import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from copy import deepcopy
from data import *
from model import *
import gc
import pickle

# samples with no targets -> diceloss???
NAME = 'GAUSSIAN_NOISE_STD_0.1'
TRAIN_FULL = False  # False: train on all 7 folds once; True: do 7-fold-cv
EPOCHS = 11
LEARNING_RATE = 1e-3


# save predictions of each valid sample here
folder_predictions = os.path.join('/home/olli/Projects/Kaggle/CryoET/Predictions', NAME)
os.makedirs(folder_predictions, exist_ok=True)
folder_weights = os.path.join('/home/olli/Projects/Kaggle/CryoET/Weights', NAME)
os.makedirs(folder_weights, exist_ok=True)

torch.set_float32_matmul_precision('medium')

valid_dice_values = []

# loop over all 7 samples and perform 7-fold-cv
for fold, sample in enumerate(samples):

    valid_samples = [sample]
    train_samples = deepcopy(samples)

    if TRAIN_FULL != True:

        print(f'============================================================= Fold {fold + 1} / {len(samples)} =============================================================')
        
        train_samples.remove(sample) # when performing cv drop the validation fold

    else:
        NAME = 'Trained_FULL_' + NAME


    data_module = LightningData(
        train_samples=train_samples,
        valid_samples=valid_samples,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # call setup explicitly; this creates the datasets which is needed to assign loss weights in NN
    data_module.setup(stage="fit")
    
    nn = NN(learning_rate=LEARNING_RATE, weights_cross_entropy=data_module.ds_train.weights_cross_entropy, stride=STRIDE, patch_size=PATCH_SIZE)
    
    logger = TensorBoardLogger(
        save_dir='/home/olli/Projects/Kaggle/CryoET/tb_logs',
        name=NAME,
        version=f'Fold_{fold}'
        )
    
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        precision='16-mixed',
        logger=logger,
        max_epochs=EPOCHS,
        #callbacks=[lr_monitor],
    )

    trainer.fit(model=nn, datamodule=data_module)

    valid_dice_values.append(nn.valid_dice_weighted)

    print(f'Valid Dice Score: {nn.valid_dice_weighted}')

    # now save the prediction from the validation sample
    tuple_save = (nn.valid_pred_volume, nn.valid_target_volume, nn.start_z, nn.start_xy)
    path_save = os.path.join(folder_predictions, f'{sample}.pickle')

    with open(path_save, 'wb') as f:
        pickle.dump(tuple_save, f)

    # save weights
    weights_filename = f'{NAME}_Fold_{fold}.pth'
    path_weights = os.path.join(folder_weights, weights_filename)

    nn.cnn.to('cpu')
    torch.save(nn.cnn.state_dict(), path_weights)

    del tuple_save
    del trainer
    del nn
    del data_module

    gc.collect()
    torch.cuda.empty_cache()
    
    if TRAIN_FULL:
        break


cv_score = sum(valid_dice_values) / len(valid_dice_values)

print(f'Final CV-Score: {cv_score}')