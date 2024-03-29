## Nhật's Hopeless Pose Restoration Challenge 2022

### Introduction:

~~You know the situation.~~ \

**New update**: The winner will be announced shortly.

### Price: 1.000.000 VND

_(Can be raised up to x5 based on how desperate I am)_

### Rules:

##### Dataset: **Human3.6M**

First,
download [data_2d_h36m_gt.npz](https://drive.google.com/file/d/1Ceg319Fpj5ZM_kSjQttITz_r2UNgddmX/view?usp=sharing) and
save it to folder `data`.

##### Objective:

- Train a model for two tasks:
    - **Identity reconstruction**: Inputs a 2D human pose of shape `[N, C, T, V]=[N, 2, 1, 17]` and outputs the same 2D
      human pose.
    - **Missing keypoints prediction**: Inputs a corrupted 2D human pose (with some keypoints randomly set to 0) and reconstructs the groundtruth 2D human
      pose.
- Free to use any normalization/augmentation technique.
- Evaluation metric:
    - `mpjpe`: mean per joint position error

##### Timeline:

_May 20th 2022 - May 23th 2022_

#### Resources:

- Example training scripts using simple overcomplete AE/VAE used in the paper _Filling the Gaps Predicting Missing Joints of Human Poses
  Using Denoising Autoencoders_.

- Visualize dataset:

```terminal
python check_data.py
```

- Refs:
  - https://stats.stackexchange.com/questions/424558/why-is-it-hard-for-a-neural-network-to-learn-the-identity-function

_All hope is lost, all light is dimmed!_
