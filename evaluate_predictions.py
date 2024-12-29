import cc3d
import pandas as pd
import torch
import numpy as np
import os
import pickle
from tqdm import tqdm
import zarr
import json
from scipy.ndimage import center_of_mass

'''BREAK THIN STRUCTURES TO SPLIT PREDICTIONS THAT ARE CLOSE TOGETHER'''


CONNECTIVITY = 26  # 6, 18 or 26  # lower means more detections



NAME_RUN = 'ACTUALLY_SMALLRADIUS_LR1e-3_20EPOCHS_96PATCHSIZE_0.1RATIOLOSS_4BS_CHANNELS_32_64_128_256_512_No_CE_Weight_Ratio_0.5'

folder_predictions = os.path.join('/home/olli/Projects/Kaggle/CryoET/Predictions', NAME_RUN)
folder_data = '/home/olli/Projects/Kaggle/CryoET/Data/train'
folder_volumes = os.path.join(folder_data, 'static', 'ExperimentRuns')
folder_labels = os.path.join(folder_data, 'overlay', 'ExperimentRuns')

num_to_class = {
    1: 'apo-ferritin',
    2: 'beta-galactosidase',
    3: 'ribosome',
    4: 'thyroglobulin',
    5: 'virus-like-particle'
}

class_radius_eval = {
    1: 60 / 10,
    2: 90 / 10,
    3: 150 / 10,
    4: 130 / 10,
    5: 135 / 10,
}

df_cols = ['id', 'experiment', 'particle_type', 'x', 'y', 'z']
df_targets = pd.DataFrame({col: [] for col in df_cols})
df_preds = pd.DataFrame({col: [] for col in df_cols})

# add class predictions and ground truths here
preds = {i: [] for i in range(1, 6)}
sample_names = []

# prediction coordinates after masks are concerted to concrete keypoints
preds_coords = {i: [] for i in range(1, 6)}

prediction_files = os.listdir(folder_predictions)

print('Loading all predictions and creating solution df')

id_targets, id_preds = 0, 0

# loop over all samples and all classes to get the targets and predictions of each class
for filename in tqdm(prediction_files):

    # get the sample name from filename
    sample = filename.split('.')[0]
    sample_names.append(sample)

    # load the volume to get the scale values
    path_volume = os.path.join(folder_volumes, sample, 'VoxelSpacing10.000', 'denoised.zarr')
    volume = zarr.open(path_volume, mode='r')
    scales = volume.attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']
    scale_z, scale_y, scale_x = scales
    del volume
    
    # low loop over all class names, load the labels from their json file and add them to the solution df
    path_labels = os.path.join(folder_labels, sample, 'Picks')

    for class_name in ['apo-ferritin', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle']:

        path_json = os.path.join(path_labels, class_name + '.json')

        with open(path_json, 'r') as f:
            label = json.load(f)

        # loop over all targets specified
        for location in label['points']:

            loc_values = location['location']

            # scale the values
            loc_x = loc_values['x']
            loc_y = loc_values['y']
            loc_z = loc_values['z']

            # create a new df with the current target and add it to the solution df
            values_df_target = (id_targets, sample, class_name, loc_x, loc_y, loc_z)
            df_target_new = pd.DataFrame({col_name: [col_value] for col_name, col_value in zip(df_cols, values_df_target)})
            df_targets = pd.concat([df_targets, df_target_new], axis=0)

            id_targets += 1

    # load the predictions            
    path_filename = os.path.join(folder_predictions, filename)

    with open(path_filename, 'rb') as f:
        (pred, target, start_z, start_xy) = pickle.load(f)

    # crop out the original size without the added borders
    pred = pred[:, start_z: start_z + 184, start_xy: start_xy + 630, start_xy: start_xy + 630]  # now shape (184, 630, 630)
    target = target[:, start_z: start_z + 184, start_xy: start_xy + 630, start_xy: start_xy + 630]

    # assign predictions
    pred = pred.argmax(0)

    # loop over all 5 classes and assign predictions and target as bool to save RAM
    for c in range(1, 6):
        
        current_target = target[c, :, :, :].numpy().astype(bool)  # dim 0 is background

        current_pred = pred == c  # filter them
        current_pred = current_pred.numpy().astype(bool)

        preds[c].append(current_pred)

    # now also load the 
    break

df_targets['id'] = df_targets['id'].astype(int)

print('Finding Connected Components and creating submission df')

for pred, sample in zip()
#labels = cc3d.connected_components(preds[1][0], connectivity=CONNECTIVITY)
#print(type(labels))
#print(labels.dtype)
#print(labels.shape)
#print(labels.sum())
#print(np.unique(labels))