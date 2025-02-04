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
import concurrent.futures
from competition_metric import *
from data import class_num_radius
import math


'''BREAK THIN STRUCTURES TO SPLIT PREDICTIONS THAT ARE CLOSE TOGETHER - Distance-Based Splitting Using Watershed
Remove large components too?
FILTER ON OTHER METRICS!! (aspect ratio or intensity values)'''

NAME_RUN = 'LR_Scheduler_Best_PATCH_SIZE_128_FocalDiceLoss'

CONNECTIVITY = 26  # 6, 18 or 26  # lower means more detections
RESIZE_FACTOR_CC = 0.5
FRACTION_VOL_CORRECT = 0.5  # if a connected component is e.g. half of a mask from that class discard the prediction


def sphere_voxels(radius):
    return int(math.pi * 4 / 3 * (radius * RESIZE_FACTOR_CC)**3)

# for cc3d.dust(labels, threshold=min_size)
thresholds = {c: sphere_voxels(class_num_radius[c] * FRACTION_VOL_CORRECT) for c in range(1, 6)}

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

df_cols = ['id', 'experiment', 'particle_type', 'x', 'y', 'z']
df_targets = pd.DataFrame({col: [] for col in df_cols})
df_preds = pd.DataFrame({col: [] for col in df_cols})

# add class predictions and ground truths here
preds = {i: [] for i in range(1, 6)}
sample_names = []
scale_values = []

# prediction coordinates after masks are concerted to concrete keypoints
preds_coords = {i: [] for i in range(1, 6)}

prediction_files = os.listdir(folder_predictions)

print('Loading all predictions and creating solution df')

id_targets = 0

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
    
    scale_values.append((scale_z, scale_y, scale_x))

    # now loop over all class names, load the labels from their json file and add them to the solution df
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
        (pred, _, start_z, start_xy) = pickle.load(f)  # target is not needed

    # crop out the original size without the added borders
    pred = pred[:, start_z: start_z + 184, start_xy: start_xy + 630, start_xy: start_xy + 630]  # now shape (184, 630, 630)

    # get class predictions
    pred = torch.softmax(pred, dim=0)
    pred = torch.argmax(pred, dim=0).type(torch.int64)  # assign classes to max prob
    pred = torch.nn.functional.one_hot(pred, num_classes=6)
    pred = pred.permute(3, 0, 1, 2)  # onehot has adds channel in last dim
    
    # now resize the predictions to speed up cc3d calculations
    pred = pred.type(torch.float32).unsqueeze(0)
    pred = torch.nn.functional.interpolate(pred, scale_factor=RESIZE_FACTOR_CC, mode='trilinear', align_corners=False)  # now shape (92, 315, 315)
    pred = pred.squeeze(0).numpy() > 0.5

    # loop over all 5 classes and assign predictions
    for c in range(1, 6): # dim 0 is background

        preds[c].append(pred[c, :, :, :])  # current class channel

    

df_targets['id'] = df_targets['id'].astype(int)

print('Finding Connected Components and creating submission df')

# takes as input connected_components with unique values and one value to find the center_of_mass
# this way all values can be processed in parallel
def get_all_components(array_unique_values, unique_value, class_name, sample, scale_z, scale_y, scale_x):

    binary_mask_component = array_unique_values == unique_value
    
    centroid = center_of_mass(binary_mask_component)
    
    z_coord, y_coord, x_coord = centroid

    z_coord *= scale_z / RESIZE_FACTOR_CC # scale them
    y_coord *= scale_y / RESIZE_FACTOR_CC
    x_coord *= scale_x / RESIZE_FACTOR_CC
    
    return (0, sample, class_name, x_coord, y_coord, z_coord)

results = []

# loop over all classes and get predictions for each sample
for c in tqdm(range(1, 6)):
    
    print('Pocessing class ', c)

    class_name = num_to_class[c]

    # loop over all samples and the predictions for this class
    for index in range(len(sample_names)):

        print(index + 1, ' / ', len(sample_names))

        sample = sample_names[index]

        scale_z, scale_y, scale_x = scale_values[index]

        # pred for this class and this sample
        pred_class_sample = preds[c][index]
        
        # find connected components; this is an array with the same shape that has unique values for each component (0 is background)
        pred_class_sample_components = cc3d.connected_components(pred_class_sample, connectivity=CONNECTIVITY)

        #print(f'\nCLASS: {c} - TH: {thresholds[c]}\n')
        #for value in np.unique(pred_class_sample_components):
        #    if value != 0:
        #        tmp = pred_class_sample_components == value
        #        print(tmp.sum())

        # filter out components that are smaller the the defined threshold
        pred_class_sample_components = cc3d.dust(pred_class_sample_components, threshold=thresholds[c])

        # get unique values and remove background class
        unique_values = np.unique(pred_class_sample_components)
        unique_values = unique_values[unique_values != 0]
        
        # create function inputs
        args = [(pred_class_sample_components, unique_value, class_name, sample, scale_z, scale_y, scale_x) for unique_value in unique_values]

        # process the unique values in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:

            result = list(executor.map(get_all_components, *zip(*args)))
        
        # collect results
        results.extend(result)

# add all predictions to a pred df
for values_df_pred in results:

    df_pred_new = pd.DataFrame({col_name: [col_value] for col_name, col_value in zip(df_cols, values_df_pred)})
    df_preds = pd.concat([df_preds, df_pred_new], axis=0)

df_preds['id'] = list(range(len(df_preds)))  # make it unique ids

print('Computing final score: ')

# compute final score from comp metric
final_score = score(
    solution=df_targets,
    submission=df_preds,
    row_id_column_name='id',
    distance_multiplier=0.5,  # ain 0.5 of  radius??
    beta=4
)

print(f'FINAL SCORE: {final_score}')