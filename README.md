## Nhật's Hopeless Pose Restoration Challenge 2022

### Introduction:

You know the situation.

### Price: 1.000.000 VND

_(Can be raised up to x5 based on how desperate I am)_

### Rules:

##### Dataset: **Human3.6M**

First,
download [data_2d_h36m_gt.npz](https://drive.google.com/file/d/1Ceg319Fpj5ZM_kSjQttITz_r2UNgddmX/view?usp=sharing) and
save it to folder `data`.

##### Objective:

- Train a model that:
    - Inputs a 2D human pose of shape `[N, C, T, V]=[N, 2, 1, 17]` and outputs the same 2D
      human pose.
    - Inputs a corrupted 2D human pose (with some keypoints randomly set to 0) and reconstructs the groundtruth 2D human
      pose.
- Free to use any normalization/augmentation technique.
- Evaluation metric:
    - `mpjpe`: mean per joint position error

##### Timeline:

_May 20th 2022 - May 23th 2022_

#### Resources:

- Example training script using simple VAE used in the paper _Filling the Gaps Predicting Missing Joints of Human Poses
  Using Denoising Autoencoders_.

```terminal
python train.py
```

- Visualize dataset:

```terminal
python check_data.py
```

_All hope is lost, all light is dimmed!_
