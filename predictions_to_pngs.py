import matplotlib.pyplot as plt
import torch
import os
import pickle
import numpy as np
import zarr
from tqdm import tqdm

NAME_RUN = 'PATCHES_CORRECT_Flipping_ZYX_Rotate_Z_IntensityNoise_Radius_Apo_5'

folder_predictions = os.path.join('/home/olli/Projects/Kaggle/CryoET/Predictions', NAME_RUN)
folder_data = '/home/olli/Projects/Kaggle/CryoET/Data/train'
folder_volumes = os.path.join(folder_data, 'static', 'ExperimentRuns')
folder_labels = os.path.join(folder_data, 'overlay', 'ExperimentRuns')

folder_save = os.path.join('/home/olli/Projects/Kaggle/CryoET/Prediction_Plots', NAME_RUN)

# colors for plotting the predictions
colors = {
    0: (255, 255, 255),  # background - white
    1: (255, 255, 0),  # apo-ferritin - yellow
    2: (255, 0, 0),  # beta-galactosidase - red
    3: (0, 255, 0),  # ribosome - green
    4: (255, 0, 255),  # thyroglobulin - pink
    5: (0, 255, 255)  # virus-like-particle - turquoise
}

prediction_files = os.listdir(folder_predictions)

# loop over all saved sample predictions, create a plot for all z_slices and save the images
for filename in prediction_files:

    print(f'Processing {filename}')

    sample = filename.split('.')[0]  # TS_5_4, ...

    path_filename = os.path.join(folder_predictions, filename)

    # load the predictions
    with open(path_filename, 'rb') as f:
        (pred, target, start_z, start_xy) = pickle.load(f)

    # crop out the original size without the added borders
    pred = pred[:, start_z: start_z + 184, start_xy: start_xy + 630, start_xy: start_xy + 630]  # now shape (184, 630, 630)
    target = target[:, start_z: start_z + 184, start_xy: start_xy + 630, start_xy: start_xy + 630]

    # load the volume
    path_volume = os.path.join(folder_volumes, sample, 'VoxelSpacing10.000', 'denoised.zarr')

    volume = zarr.open(path_volume, mode='r')

    # add the high res array to the dict
    volume = volume[0]

    # folder to save the images to
    folder_save_sample = os.path.join(folder_save, sample)
    os.makedirs(folder_save_sample, exist_ok=True)

    # loop over all z_slices and create a plot with the input, the prediction and the target
    for index_z in tqdm(range(0, 184)):

        # first get the target
        current_target = target[:, index_z, :, :].type(torch.int8)

        current_target = current_target.argmax(0)

        current_target_color = torch.zeros((630, 630, 3), dtype=torch.uint8)

        for key, color in colors.items():
            mask = current_target == key
            current_target_color[mask] = torch.tensor(color, dtype=torch.uint8)   

        current_target_color = current_target_color.numpy().astype(np.uint8)

        # now create the prediction
        current_pred = pred[:, index_z, :, :]

        current_pred = current_pred.argmax(0)

        current_pred_color = torch.zeros((630, 630, 3), dtype=torch.uint8)

        for key, color in colors.items():
            mask = current_pred == key
            current_pred_color[mask] = torch.tensor(color, dtype=torch.uint8)
        
        current_pred_color = current_pred_color.numpy().astype(np.uint8)

        slice_volume = volume[index_z, :, :]

        fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
        ax[0].imshow(slice_volume, cmap='gray')
        ax[1].imshow(current_target_color)
        ax[2].imshow(current_pred_color)
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')

        fig.suptitle(f'Sample: {sample}; Z-Slice: {index_z + 1} / 184')
        fig.tight_layout()
        fig.savefig(f'{folder_save_sample}/{str(index_z)}.png')

        plt.close(fig)