# CZII - CryoET Object Identification

## 98th place solution: https://www.kaggle.com/competitions/czii-cryo-et-object-identification/overview

## Approach: Single 3D-UNet to detect segmentation masks & run connected-components-3d to localize detections

![Description](75.png)

- 7-fold-CV: Train on patches from 6 samples  
- Validate on patches from hold-out sample  
- Save predictions from hold-out fold to disc
- After 7-folds run competition metric on all predictions at once to get final score

### CV-score: 0.7781; Public-LB: 0.72782; Private-LB: 0.71783
