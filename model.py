import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import monai
from monai.networks.nets.segresnet import SegResNet as SRN
from torchinfo import summary
from monai.losses import DiceLoss

'''Weights DiceLoss!
Punish overlap in loss?'''
RATIO_LOSSES = 0.5  # 0-1 (weight for crossentropy vs diceloss)
INCLUDE_BACCKGROUND_DICELOSS = False  # if the background class should be included for dice loss calculation

DROP_RATE = 0.2
CHANNELS = (32, 64, 128, 256, 512)  # (32, 64, 128, 256, 512)
STRIDES = (2, 2, 1, 1)  # (2, 2, 2, 2)

assert len(CHANNELS) == len(STRIDES) + 1



class NN(pl.LightningModule):

    def __init__(
            self,
            learning_rate,
            weights_cross_entropy,
            stride,
            patch_size,
            ratio_losses=RATIO_LOSSES,
            include_background_diceloss=INCLUDE_BACCKGROUND_DICELOSS,
            drop_rate=DROP_RATE,
            channels=CHANNELS,
            strides=STRIDES,
            ):

        super().__init__()

        self.learning_rate = learning_rate
        self.weights_cross_entropy = weights_cross_entropy.to('cuda:0')
        self.ratio_losses = ratio_losses
        self.diceloss = DiceLoss(include_background=include_background_diceloss, softmax=False, reduction='mean')
        self.mask_loss, self.ignore_size = self.create_mask_loss(stride=stride, patch_size=patch_size)

        self.drop_rate=drop_rate
        self.channels=channels
        self.strides=strides

        self.cnn = monai.networks.nets.UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=6,
            channels=self.channels,
            strides=self.strides,
            dropout=self.drop_rate,
        )

        self.valid_dice = []

        # assign predictions of last epoch here
        self.valid_pred_volume = None
        self.valid_target_volume = None

        self.start_z = None
        self.start_xy = None

    def forward(self, x):
        
        pred = self.cnn(x)

        return pred
    
    # takes as input the patch_size and stride and caculated the mask such that every region is used once
    def create_mask_loss(self, stride, patch_size):

        # mask should have all zeros except for the inner cube thats relevant
        mask = torch.zeros((patch_size, patch_size, patch_size))

        # calculate the inner cude such that with the defined stride allways contributes onces
        ignore_size = int(patch_size * (1 - stride) / 2)
        
        mask[ignore_size: -ignore_size, ignore_size: -ignore_size, ignore_size: -ignore_size] = 1

        mask = mask.to('cuda:0')

        return mask, ignore_size

    
    # takes as input prediction probabilities and targets
    # calculates the IoU for all 6 classes and the average from them
    def calculate_IoU_Dice_scores(self, preds_sm, targets):

        with torch.no_grad():

            preds_= preds_sm.detach()
            y = targets.detach().type(torch.int64)

            # crop out the inner cube of relevante
            preds_ = preds_[:, :, self.ignore_size:-self.ignore_size, self.ignore_size:-self.ignore_size, self.ignore_size:-self.ignore_size]
            y = y[:, :, self.ignore_size:-self.ignore_size, self.ignore_size:-self.ignore_size, self.ignore_size:-self.ignore_size]

            # first assign classes to prodictions
            preds_ = torch.argmax(preds_, dim=1).type(torch.int64)

            # now create onehot class predictions
            preds_ = F.one_hot(preds_, num_classes=6).permute(0, 4, 1, 2, 3)  # put class dim in second place again

            iou_scores = []
            dice_scores = []

            for c in range(6):

                pred_class = preds_[:, c]  # class predictions for the batch (bs, z, y, x)
                target_class = y[:, c]  # targets for the batch (bs, z, y, x)

                # calculations for IoU
                intersection = (pred_class & target_class).sum(dim=(1, 2, 3))  # per-batch intersection
                union = (pred_class | target_class).sum(dim=(1, 2, 3))  # Per-batch union

                iou = (intersection / (union + 1e-6)).mean()  # iou for the current class and batch
                iou_scores.append(iou.item())

                # calculations for dice score
                pred_sum = pred_class.sum(dim=(1, 2, 3))  # sum of predicted positives
                target_sum = target_class.sum(dim=(1, 2, 3))  # sum of target positives
                dice = (2 * intersection / (pred_sum + target_sum + 1e-6)).mean()  # dice for current class and batch
                dice_scores.append(dice.item())

            # average the iou scores for all classes
            mean_iou = sum(iou_scores) / len(iou_scores)
            mean_iou_no_background = sum(iou_scores[1:]) / len(iou_scores[1:])

            # average the dice_scores for all classes
            mean_dice = sum(dice_scores) / len(dice_scores)
            mean_dice_no_background = sum(dice_scores[1:]) / len(dice_scores[1:])
            
            iou_background, iou_apo_ferritin, iou_beta_galactosidase, iou_ribosome, iou_thyroglobulin, iou_virus = iou_scores
            dice_background, dice_apo_ferritin, dice_beta_galactosidase, dice_ribosome, dice_thyroglobulin, dice_virus = dice_scores

            return mean_iou, mean_iou_no_background, iou_background, iou_apo_ferritin, iou_beta_galactosidase, iou_ribosome, iou_thyroglobulin, iou_virus, mean_dice, mean_dice_no_background, dice_background, dice_apo_ferritin, dice_beta_galactosidase, dice_ribosome, dice_thyroglobulin, dice_virus


    def training_step(self, batch, batch_ix):

        inputs, targets, _ = batch

        preds = self.forward(inputs)

        # add the batch size to the mask; currently shape (patch_size, patch_size, patch_size)
        batch_size = preds.size()[0]
        mask = self.mask_loss.clone()
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # shape (batch_size, patch_size, patch_size, patch_size)

        # mask the preds and targets, create integer classes instead of onehots, calculate loss and divide it by the number of valid voxels
        masked_targets = torch.argmax(targets, dim=1) * mask
        masked_preds = preds * mask.unsqueeze(1)  # add channel dim
        
        # cross entropy loss wont work with onehot targets
        cross_entropy_loss = F.cross_entropy(masked_preds, masked_targets.long(), reduction='sum')  # , weight=self.weights_cross_entropy, reduction='sum')
        cross_entropy_loss = cross_entropy_loss / mask.sum()  # normalize the loss
        
        # dice expects softmax not logits and onehot targets
        preds_sm = F.softmax(preds, dim=1)

        # also apply mask
        masked_preds_sm = preds_sm * mask.unsqueeze(1)
        masked_targets_onehot = targets * mask.unsqueeze(1)
        
        dice_loss = self.diceloss(masked_preds_sm, masked_targets_onehot)

        total_loss = cross_entropy_loss * self.ratio_losses + dice_loss * (1 - self.ratio_losses)
        
        mean_iou, mean_iou_no_background, iou_background, iou_apo_ferritin, iou_beta_galactosidase, iou_ribosome, iou_thyroglobulin, iou_virus, mean_dice, mean_dice_no_background, dice_background, dice_apo_ferritin, dice_beta_galactosidase, dice_ribosome, dice_thyroglobulin, dice_virus = self.calculate_IoU_Dice_scores(preds_sm, targets)

        dict_logger = {
            'Train_Total_Loss': total_loss,
            'Train_Cross_Entropy_Weighted': cross_entropy_loss,
            'Train_Dice_Loss': dice_loss,
            'Train_IoU': mean_iou,
            'Train_IoU_No_Background': mean_iou_no_background,
            'Train_IoU_Background': iou_background,
            'Train_IoU_Apo_Ferritin': iou_apo_ferritin,
            'Train_IoU_Beta_Galactosidase': iou_beta_galactosidase,
            'Train_IoU_Ribosome': iou_ribosome,
            'Train_IoU_Thyroglobulin': iou_thyroglobulin,
            'Train_IoU_Virus': iou_virus,
            'Train_Dice': mean_dice,
            'Train_Dice_No_Background': mean_dice_no_background,
            'Train_Dice_Background': dice_background,
            'Train_Dice_Apo_Ferritin': dice_apo_ferritin,
            'Train_Dice_Beta_Galactosidase': dice_beta_galactosidase,
            'Train_Dice_Ribosome': dice_ribosome,
            'Train_Dice_Thyroglobulin': dice_thyroglobulin,
            'Train_Dice_Virus': dice_virus,
        }

        self.log_dict(dict_logger, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return total_loss
    
    def validation_step(self, batch, batch_ix):

        inputs, targets, patch_coords = batch

        preds = self.forward(inputs)

        # add the batch size to the mask; currently shape (patch_size, patch_size, patch_size)
        batch_size = preds.size()[0]
        mask = self.mask_loss.clone()
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # shape (batch_size, patch_size, patch_size, patch_size)

        # mask the preds and targets, create integer classes instead of onehots, calculate loss and divide it by the number of valid voxels
        masked_targets = torch.argmax(targets, dim=1) * mask
        masked_preds = preds * mask.unsqueeze(1)  # add channel dim

        # cross entropy loss wont work with onehot targets
        cross_entropy_loss = F.cross_entropy(masked_preds, masked_targets.long(), reduction='sum')  # , weight=self.weights_cross_entropy, reduction='sum')
        cross_entropy_loss = cross_entropy_loss / mask.sum()  # normalize the loss

        # dice expects softmax not logits and onehot targets
        preds_sm = F.softmax(preds, dim=1)

        # also apply mask
        masked_preds_sm = preds_sm * mask.unsqueeze(1)
        masked_targets_onehot = targets * mask.unsqueeze(1)

        dice_loss = self.diceloss(masked_preds_sm, masked_targets_onehot)

        total_loss = cross_entropy_loss * self.ratio_losses + dice_loss * (1 - self.ratio_losses)

        mean_iou, mean_iou_no_background, iou_background, iou_apo_ferritin, iou_beta_galactosidase, iou_ribosome, iou_thyroglobulin, iou_virus, mean_dice, mean_dice_no_background, dice_background, dice_apo_ferritin, dice_beta_galactosidase, dice_ribosome, dice_thyroglobulin, dice_virus = self.calculate_IoU_Dice_scores(preds_sm, targets)

        dict_logger = {
            'Valid_Total_Loss': total_loss,
            'Valid_Cross_Entropy_Weighted': cross_entropy_loss,
            'Valid_Dice_Loss': dice_loss,
            'Valid_IoU': mean_iou,
            'Valid_IoU_No_Background': mean_iou_no_background,
            'Valid_IoU_Background': iou_background,
            'Valid_IoU_Apo_Ferritin': iou_apo_ferritin,
            'Valid_IoU_Beta_Galactosidase': iou_beta_galactosidase,
            'Valid_IoU_Ribosome': iou_ribosome,
            'Valid_IoU_Thyroglobulin': iou_thyroglobulin,
            'Valid_IoU_Virus': iou_virus,
            'Valid_Dice': mean_dice,
            'Valid_Dice_No_Background': mean_dice_no_background,
            'Valid_Dice_Background': dice_background,
            'Valid_Dice_Apo_Ferritin': dice_apo_ferritin,
            'Valid_Dice_Beta_Galactosidase': dice_beta_galactosidase,
            'Valid_Dice_Ribosome': dice_ribosome,
            'Valid_Dice_Thyroglobulin': dice_thyroglobulin,
            'Valid_Dice_Virus': dice_virus,
        }
        self.log_dict(dict_logger, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        # calculate mean_iou for all batches in last epoch; also assign the predictions
        if self.current_epoch == self.trainer.max_epochs - 1:
            self.valid_dice.append(mean_dice)

            patch_coords = patch_coords.cpu().numpy()  # shape (bs, 3)
            
            patch_size = preds_sm.size()[-1]

            for i in range(patch_coords.shape[0]):

                z, y, x = patch_coords[i, :]

                z = int(z)
                y = int(y)
                x = int(x)
                
                # assign the prediction and targets at the lication; do this by adding the masked pred/targets in the full volume; this way only the middle cube is used
                self.valid_pred_volume[:, z: z + patch_size, y: y + patch_size, x: x + patch_size] += masked_preds_sm[i, :, :, :, :].cpu()  # adding them s
                self.valid_target_volume[:, z: z + patch_size, y: y + patch_size, x: x + patch_size] += masked_targets_onehot[i, :, :, :, :].cpu().type(torch.bool)

        return total_loss
    
    def on_validation_epoch_start(self):
        
        # initialize it again because during sanity checking it changes..
        if self.current_epoch == self.trainer.max_epochs - 1:
            self.valid_dice = []

            # assign predictions of last epoch here; trainer must be connected which is why one cannot init it with it
            self.valid_pred_volume = torch.zeros(size=(6, self.trainer.datamodule.ds_valid.shape_z, self.trainer.datamodule.ds_valid.shape_xy, self.trainer.datamodule.ds_valid.shape_xy)).type(torch.float32)  # shape (n_classes, dim_z, dim_xy, dim_xy)
            self.valid_target_volume = torch.zeros(size=(6, self.trainer.datamodule.ds_valid.shape_z, self.trainer.datamodule.ds_valid.shape_xy, self.trainer.datamodule.ds_valid.shape_xy)).type(torch.bool)

            self.start_z = self.trainer.datamodule.ds_valid.start_z
            self.start_xy = self.trainer.datamodule.ds_valid.start_xy
    
    def on_validation_epoch_end(self):
        
        if self.current_epoch == self.trainer.max_epochs - 1:

            self.valid_dice = sum(self.valid_dice) / len(self.valid_dice)
    
    def configure_optimizers(self):
        
        optim = torch.optim.Adam(self.cnn.parameters(), lr=self.learning_rate)

        return optim
    


        
if __name__ == '__main__':

    '''cnn = SRN(
        spatial_dims=3,  # 3d inputs
        init_filters=32,  # filters of first conv layer
        in_channels=1,  # one channel inputs
        out_channels=6,
        dropout_prob=DROP_RATE,
        blocks_down=BLOCKS_DOWN,
        blocks_up=BLOCKS_UP,
    )'''

    cnn = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=6,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        dropout=DROP_RATE
    )

    exp_input = torch.zeros(2, 1, 48, 48, 48)
    print(summary(cnn, input_data=exp_input))
    print(cnn(exp_input).size())
    with torch.no_grad():
        pred = cnn(exp_input)
    pred = F.softmax(pred, dim=1)
    for batch_ix in range(2):
        for dim in range(5):
            print(pred[batch_ix, dim, :, :, :].min(), pred[batch_ix, dim, :, :, :].mean(), pred[batch_ix, dim, :, :, :].max())

