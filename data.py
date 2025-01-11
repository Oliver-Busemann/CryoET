import torch
import pytorch_lightning as pl
import os
from glob import glob
import zarr
import json
import numpy as np
import matplotlib.pyplot as plt
import random
from monai.transforms import Compose, RandFlipd, RandRotated, RandAffine
import math

''''
X, Y, Z coordinates have relevance to pdata.output_coordinatesrobability of each target???
UPSAMPLE IN FREQUENCY
'''

PATCH_SIZE = 96  # size of the 3d patches to crop out; only calculate loss for the inner cube; use a mask for this and a fitting stride such that each region contributes once
STRIDE = 0.75  # stride when cropping out patches ()
assert (1 - STRIDE) * PATCH_SIZE % 2 == 0
BATCH_SIZE = 4
NUM_WORKERS=16
PIN_MEMORY=False

folder_data = '/home/olli/Projects/Kaggle/CryoET/Data/train'
folder_volumes = os.path.join(folder_data, 'static', 'ExperimentRuns')
folder_labels = os.path.join(folder_data, 'overlay', 'ExperimentRuns')

samples = os.listdir(folder_volumes)  # name of the 7 samples (TS_5_4, ...)

volumes = {}  # load all volumes and add them to the dict

labels = {}  # same for the labels

# 5 classes
classes = {
    'apo-ferritin': 1,
    'beta-galactosidase': 2,
    'ribosome': 3,
    'thyroglobulin': 4,
    'virus-like-particle': 5,
}

# weights for computing the metric
class_weights = {
    'apo-ferritin': 1,
    'beta-galactosidase': 2,
    'ribosome': 1,
    'thyroglobulin': 2,
    'virus-like-particle': 1,
}

# radius for a correct prediction for each class (10A scale)
class_radiuses = {
    'apo-ferritin': 60 / 10,
    'beta-galactosidase': 90 / 10,
    'ribosome': 150 / 10,
    'thyroglobulin': 130 / 10,
    'virus-like-particle': 135 / 10,
}

#class_num_radius = {
#    1: 60 / 10,
#    2: 90 / 10,
#    3: 150 / 10,
#    4: 130 / 10,
#    5: 135 / 10,
#}

class_num_radius = {
    1: 6,
    2: 6,
    3: 10,
    4: 6,
    5: 12,
}

# loop over all samples to create volumes and its labels
for sample in samples:

    # load the volume
    path_volume = os.path.join(folder_volumes, sample, 'VoxelSpacing10.000', 'denoised.zarr')

    volume = zarr.open(path_volume, mode='r')

    # add the high res array to the dict
    volumes[sample] = volume[0]

    # get the scales to transform labels
    scales = volume.attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']
    scale_z, scale_y, scale_x = scales

    path_labels = os.path.join(folder_labels, sample, 'Picks')

    # add each label as a tuple to a list for that sample (x, y, z, class)
    labels[sample] = []

    # each class has its labels in one json
    for class_name in classes.keys():
        path_json = os.path.join(path_labels, class_name + '.json')

        with open(path_json, 'r') as f:
            label = json.load(f)

        # loop over all targets specified
        for location in label['points']:

            loc_values = location['location']

            # scale the values
            loc_x = loc_values['x'] / scale_x
            loc_y = loc_values['y'] / scale_y
            loc_z = loc_values['z'] / scale_z

            # add a tuple to the labels
            labels[sample].append((loc_x, loc_y, loc_z, classes[class_name]))

# augmentations
transform = Compose([
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),  # z
    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),  # y
    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),  # x
    RandRotated(
        keys=["image", "label"],
        range_x=(-math.pi, math.pi),  # this is actually z dim so rotate only the images essentially
        range_y=0,
        range_z=0,
        prob=1,
        mode=["bilinear", "nearest"],
        padding_mode='zeros'
    ),
])

# create a dataset that takes as input the samples to use (e.g. ['TS_6_6', 'TS_99_9', 'TS_73_6', 'TS_86_3', 'TS_6_4', 'TS_69_2'] for train and ['TS_5_4'] for valid)
# this way we can easily to a 7-fold-cv
# for each volume (sample) crop out patches in the defined size and stride
class Data(torch.utils.data.Dataset):

    def __init__(self, sample_names, transform=None):
        self.sample_names = sample_names
        self.transform = transform

        # crop out patches from the volumes and the target volumes and append them to lists
        self.inputs = []
        self.outputs = []

        # save z, y, x coordinates to a list to later reconstruct full volume for eval/submission
        self.patch_coords = []

        # staring coordinates where the volume is placed in to have a border with zeros
        self.start_z = 0
        self.start_xy = 0  # same for x & y
        self.shape_z = 0
        self.shape_xy = 0

        # create patches
        self.create_patches()

        # shuffle the lists
        self.shuffle_lists()

        # for equal weights in all 6 classes for cross_entropy
        self.weights_cross_entropy = torch.ones(size=(6,))

        # calculate weights before training
        self.calculate_equal_weights()

    def create_patches(self):

        # create patches for all samples
        for sample_name in self.sample_names:

            # shapes of the volumes are (184, 630, 630)
            # now find the final dimensions that make each dim perfectly dividable by the PATCH_SIZE and add half the PATCH_SIZE for the borders
            dim_z = 184 // PATCH_SIZE * PATCH_SIZE
            rest_z = 184 % PATCH_SIZE
            if rest_z < (PATCH_SIZE / 2):  # one additional PATCH_SIZE is enough
                dim_z += PATCH_SIZE
            else:
                dim_z += 2 * PATCH_SIZE

            # x-y are the same
            dim_xy = 630 // PATCH_SIZE * PATCH_SIZE
            rest_xy = 630 % PATCH_SIZE
            if rest_xy < (PATCH_SIZE / 2):
                dim_xy += PATCH_SIZE
            else:
                dim_xy += 2 * PATCH_SIZE

            # now place the volume in a larger empty one such that the boarders are added
            # calculate the starting points where to insert it first
            start_z = int((dim_z - 184) / 2)
            start_xy = int((dim_xy - 630) / 2)

            volume_large = np.zeros(shape=(dim_z, dim_xy, dim_xy), dtype=np.float32)
            volume_large[start_z: start_z + 184, start_xy: start_xy + 630, start_xy: start_xy + 630] = volumes[sample_name]

            # create the target volume with all masks (has 6 channels for 5 classes plus background)
            volume_target = np.zeros(shape=(6, dim_z, dim_xy, dim_xy), dtype=np.uint8)

            # assign each keypoint to the correct location with the corresponding class value
            for (x, y, z, c) in labels[sample_name]:
                
                # get the sphere mask of the current keypoint
                mask, size_mask = self.keypoint_to_mask((x, y, z, c))
                size_mask_lower = size_mask // 2
                size_mask_upper = (size_mask // 2) + (size_mask % 2 > 0)  # round rest up

                z_mid = start_z + int(round(z, 0))
                y_mid = start_xy + int(round(y, 0))
                x_mid = start_xy + int(round(x, 0))

                # assign the mask in the correct class channel and location; however only add the region where the values are one! else it would override other masks close to it that allready were assigned
                # get the correct region first
                roi = volume_target[c, z_mid - size_mask_lower: z_mid + size_mask_upper, y_mid - size_mask_lower: y_mid + size_mask_upper, x_mid - size_mask_lower: x_mid + size_mask_upper]
                np.logical_or(roi, mask, out=roi)
            
            # add the label for the background class
            volume_target[0] = (np.sum(volume_target[1:6], axis=0) == 0)
            
            # now crop out all possible patches in the defined stride
            for z in range(0, dim_z - PATCH_SIZE + 1, int(PATCH_SIZE * STRIDE)):
                for y in range(0, dim_xy - PATCH_SIZE + 1, int(PATCH_SIZE * STRIDE)):
                    for x in range(0, dim_xy - PATCH_SIZE + 1, int(PATCH_SIZE * STRIDE)):

                        current_input = volume_large[z: z + PATCH_SIZE, y: y + PATCH_SIZE, x: x + PATCH_SIZE]
                        current_output = volume_target[:, z: z + PATCH_SIZE, y: y + PATCH_SIZE, x: x + PATCH_SIZE]
                        
                        # append them to the lists
                        self.inputs.append(current_input)
                        self.outputs.append(current_output)
                        self.patch_coords.append([z, y, x])

        # same for all samples
        self.start_z = start_z
        self.start_xy = start_xy
        self.shape_z = dim_z
        self.shape_xy = dim_xy

    # shuffle all cropped out patches/labels/start_coord the same way
    def shuffle_lists(self):
        
        list_comb = list(zip(self.inputs, self.outputs, self.patch_coords))
        random.shuffle(list_comb)
        list1_shuffled, list2_shuffled, list3_shuffled= zip(*list_comb)

        self.inputs = list(list1_shuffled)
        self.outputs = list(list2_shuffled)
        self.patch_coords = list(list3_shuffled)

    # create a binary mask with size 2*radius for the class
    def keypoint_to_mask(self, keypoint):

        (z, y, x, c) = keypoint

        # radius for this class
        radius = int(round(class_num_radius[c], 0))

        size_mask = 2 * radius + 1

        # create grid coordinates
        z_dim, y_dim, x_dim = np.ogrid[:size_mask, :size_mask, :size_mask]

        # calculate the squared distance from the center (middle)
        center = radius  # center is at (radius, radius, radius)
        distance_squared = (z_dim - center)**2 + (y_dim - center)**2 + (x_dim - center)**2

        # mask to assign in the volume
        mask = distance_squared <= radius**2
        
        return mask, size_mask

    # loop over all targets, create masks and count targets to finally calculate equal weights
    def calculate_equal_weights(self):

        class_counts = [0, 0, 0, 0, 0, 0]
        
        for mask in self.outputs:
            
            # loop over all 6 masks and count targets
            for i in range(6):

                current_targets = mask[i, :, :, :].sum()
                class_counts[i] += current_targets

        class_counts = np.array(class_counts, dtype=np.uint64)
        
        class_weights = []

        for class_count in class_counts:
            class_weights.append(class_counts.sum() / class_count)

        # finally assign it
        self.weights_cross_entropy = torch.Tensor(class_weights).type(torch.float32)

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):

        vol = self.inputs[index]  # (z, y, x)
        segmentation_mask = self.outputs[index]  # (n_channels, z, y, x)

        # for monai augmentation to work the shapes need to match; add channel for input
        vol = np.expand_dims(vol, axis=0)

        # augment volume and targetif specified
        if self.transform is not None:

            data = {
                "image": vol,
                "label": segmentation_mask
            }

            augmented_data = self.transform(data)
            vol = augmented_data["image"]
            segmentation_mask = augmented_data["label"]

        # normalize the inputs to mean 0 std 1
        try:
            mean = vol.mean()
            std = vol.std()
            vol = (vol - mean) / std

        # if all values same...
        except ZeroDivisionError:
            pass

        # get patch coordinates (z, y, x)
        patch_coord = self.patch_coords[index]
        
        vol = torch.Tensor(vol).type(torch.float32)
        segmentation_mask = torch.Tensor(segmentation_mask).type(torch.float32)
        patch_coord = torch.Tensor(patch_coord).type(torch.float32)

        return vol, segmentation_mask, patch_coord


class LightningData(pl.LightningDataModule):

    def __init__(
        self,
        train_samples,
        valid_samples,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        transform=transform
        ):

        super().__init__()

        self.train_samples = train_samples
        self.valid_samples = valid_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = transform

    def setup(self, stage):
        
        self.ds_train = Data(sample_names=self.train_samples, transform=self.transform)
        self.ds_valid = Data(sample_names=self.valid_samples)

    def train_dataloader(self):

        dl_train = torch.utils.data.DataLoader(
            dataset=self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory            
        )

        return dl_train
    
    def val_dataloader(self):

        dl_valid = torch.utils.data.DataLoader(
            dataset=self.ds_valid,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory            
        )

        return dl_valid




if __name__ == '__main__':
    
    data = Data(sample_names=['TS_5_4'], transform=transform)  # , 'TS_6_6', 'TS_99_9', 'TS_73_6', 'TS_86_3', 'TS_6_4', 'TS_69_2'])

    _ = data.__getitem__(0)
    
    
    '''for i in range(100):
        input_, segmentation_mask, coords = data.__getitem__(i)

        input_ = input_.squeeze(0).numpy()
        segmentation_mask = segmentation_mask.numpy()

        if segmentation_mask[-1, :, :, :].max() > 0:

            for z_slice in range(PATCH_SIZE):
                fig, ax = plt.subplots(ncols=7, figsize=(3*7, 3))
                ax[0].imshow(input_[z_slice, :, :], cmap='gray')
                ax[0].axis('off')
                for c in range(1, 7):
                    
                    ax[c].imshow(segmentation_mask[c - 1, z_slice, :, :], cmap='gray', vmin=0, vmax=1)
                    ax[c].axis('off')
                plt.show()

            break'''
    '''for ix in range(0, len(data.inputs), 10):
        i, mask, _ = data.__getitem__(ix)

        i = np.array(i)
        mask = np.array(mask)

        for c in range(1, 6):
            print(c, mask[c, :, :, :].max())
            #for z_slice in range(48):

            #    print(mask[c, z_slice, :, :].max(), c)

        # now loop in parallel over all z slices from input and output and plot them next to each other
        fig, ax = plt.subplots(nrows=PATCH_SIZE, ncols=7, figsize=(7 * 2, PATCH_SIZE * 2))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        for slice in range(0, PATCH_SIZE):

            ax[slice, 0].imshow(i[0, slice, :, :], cmap='gray')
            ax[slice, 0].axis('off')

            for index in range(1, 7):
                ax[slice, index].imshow(mask[index - 1, slice, :, :], cmap='gray', vmin=0, vmax=1)
                ax[slice, index].axis('off')

        plt.tight_layout()

        fig.savefig(f'/home/olli/Downloads/Example_Seg_Mask_Background_{ix}.png')'''
