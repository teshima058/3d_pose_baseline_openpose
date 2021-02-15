# 3D Pose Baseline for OpenPose

A PyTorch implementation of a simple baseline for 3d human pose estimation from output of OpenPose.
You can check the original Tensorflow implementation written by [Julieta Martinez et al.](https://github.com/una-dinosauria/3d-pose-baseline).
Some codes for data processing are brought from the original version, thanks to the authors.

This is the code for the paper

```
@inproceedings{martinez_2017_3dbaseline,
  title={A simple yet effective baseline for 3d human pose estimation},
  author={Martinez, Julieta and Hossain, Rayat and Romero, Javier and Little, James J.},
  booktitle={ICCV},
  year={2017}
}
```

## Tested System Software
* Windows (Version 10, 64-bit)
* Python 3.7.9
* The required Python library is written in [requirements.txt](./requirements.txt)

Please run `pip install -r requirements.txt`

## Quick Start
1. Download pretrained model file [best.chkpt](https://drive.google.com/file/d/1VECM2_SA3WbwK4_vVJ0h1telcFKBJ2vm/view?usp=sharing)
1. Run `python predict.py [pretrained_model_path] [openpose_json_dir] --save_mp4 [mp4 path to save] --save_csv [csv path to save] `

      The saved csv file is the format of the data captured by Kinect.

## Training
### 1. Preparation
1. Download CMU Panoptic Dataset from http://domedb.perception.cs.cmu.edu/dataset.html
2. Place the data like this
```
${DATASET_DIR}
      |-- hdPose3d_stage1_coco19_171026_pose1
            |-- body3DScene_00000160.json
            |-- body3DScene_00000161.json
            |-- ...
      |-- hdPose3d_stage1_coco19_171026_pose2
            |-- body3DScene_00000109.json
            |-- body3DScene_00000109.json
            |-- ...
      |-- ...
```
3. Run `python generate_dataset.py [DATASET_DIR] [data_suffix] [save_dataset_path] [--data_ratio] {data_ratio}` 
and you can obtained the dataset file at `[save_dataset_path]`. `[data_suffix]` is hdPose3d_stage1 in the above example and is used when searching the folder.

      You can use [the dataset](https://drive.google.com/drive/folders/1J4sgS-XDMXZUFYrmlgRjo3H35b_46YIX?usp=sharing) that has already been created.

### 2. Training
1. Check `opt.py` and change settings such as dataset, checkpoint path, etc.

3. Run `train.py`

