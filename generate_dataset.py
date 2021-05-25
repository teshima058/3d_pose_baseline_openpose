import os
import json
import math
import pickle
from random import randint, uniform
import numpy as np
import argparse 

from pyquaternion import Quaternion
from scipy.spatial import distance
from tqdm import tqdm
import torch

from src.datasets.utils import angle_between
from src.poseVisualizer import visualizePose


cmu_joint_index = [''] * 19
cmu_joint_index[0]  = 'Neck'
cmu_joint_index[1]  = 'Nose'
cmu_joint_index[2]  = 'MidHip'
cmu_joint_index[3]  = 'LShoulder'
cmu_joint_index[4]  = 'LElbow'
cmu_joint_index[5]  = 'LWrist'
cmu_joint_index[6]  = 'LHip'
cmu_joint_index[7]  = 'LKnee'
cmu_joint_index[8]  = 'LAnkle'
cmu_joint_index[9]  = 'RShoulder'
cmu_joint_index[10] = 'RElbow'
cmu_joint_index[11] = 'RWrist'
cmu_joint_index[12] = 'RHip'
cmu_joint_index[13] = 'RKnee'
cmu_joint_index[14] = 'RAnkle'
cmu_joint_index[15] = 'LEye'
cmu_joint_index[16] = 'LEar'
cmu_joint_index[17] = 'REye'
cmu_joint_index[18] = 'REar'


def normalize_skeleton(_skel, mode='coco'):
    """
    Recoordinate the skeleton so that the position of the neck joint is (0,0) or (0,0,0).
    And normalize shoulder length to 1

    Specify 'coco' or 'cmu' for mode
    coco : _skel.shape = ({frame_num}, 19, 2) or ({frame_num}, 19, 3) 
    cmu  : _skel.shape = ({frame_num}, 18, 2) or ({frame_num}, 18, 3) 

    Keyword Arguments:
    _skel - skeleton pose, unnormalized
    """

    if mode == 'coco':
        neck_joint_idx = 1
        nose_joint_idx = 0
        r_shoulder_joint_idx = 2
        l_shoulder_joint_idx = 5
    elif mode == 'cmu' or mode == 'openpose':
        neck_joint_idx = 0
        nose_joint_idx = 1
        r_shoulder_joint_idx = 9
        l_shoulder_joint_idx = 3
    else:
        raise AssertionError("Choose 'coco' or 'cmu' for the normalization argument")
    
    if mode == 'cmu':
        new_poses = []
        angles = []
        shoulder_lengths = []
        neck_positions = []
        for pose in _skel:
            shoulder_len = distance.euclidean(pose[neck_joint_idx], pose[l_shoulder_joint_idx])
            shoulder_lengths.append(shoulder_len)
            neck_positions.append(pose[neck_joint_idx])
            
            new_pose = []
            for joint in pose:
                new_pose.append((joint - pose[neck_joint_idx]) / shoulder_len)
            new_poses.append(new_pose)
        new_poses = np.array(new_poses)

        return new_poses, np.array(shoulder_lengths), np.array(neck_positions)

    elif mode == 'openpose':
        # ----- Localization -----
        localize_pose = []
        neck_positions = []
        for pose in _skel:
            tmp = []
            for joint in pose:
                tmp.append(joint - pose[neck_joint_idx])
            localize_pose.append(tmp)
            neck_positions.append(pose[neck_joint_idx])
        localize_pose = np.array(localize_pose)
        neck_positions = np.array(neck_positions)

        # ----- Scaling ------
        # Find the frame with the speaker facing most front
        face_front_value = 10000
        for i in range(len(localize_pose)):
            # Search by x coordinate of the nose
            if face_front_value > abs(localize_pose[i][nose_joint_idx][0]):
                face_front_value = abs(localize_pose[i][nose_joint_idx][0])
                face_front_frame = i
        scale_factor = distance.euclidean(localize_pose[face_front_frame][neck_joint_idx], localize_pose[face_front_frame][l_shoulder_joint_idx])

        normalized_pose = []
        for pose in localize_pose:
            tmp = []
            for joint in pose:
                tmp.append(joint / scale_factor)
            normalized_pose.append(tmp)
        normalized_pose = np.array(normalized_pose)

        return normalized_pose, neck_positions, scale_factor
    
    else:
        print("[Error] supecify mode ('coco' or 'cmu' or 'openpose') when normalizing")
        sys.exit(-1)


def unNormalize_skeleton(_skel, shoulder_lengths, neck_positions, mode='coco'):
    if mode == 'coco':
        neck_joint_idx = 1
        r_shoulder_joint_idx = 2
        l_shoulder_joint_idx = 5
    elif mode == 'cmu':
        neck_joint_idx = 0
        r_shoulder_joint_idx = 9
        l_shoulder_joint_idx = 3
    else:
        raise AssertionError("Choose 'coco' or 'cmu' for the normalization argument")

    unnorm_poses = []
    for i,pose in enumerate(_skel):
        unnorm_pose = []
        for j,joint in enumerate(pose):
            x = joint[0] * shoulder_lengths[i] + neck_positions[i][0]
            y = joint[1] * shoulder_lengths[i] + neck_positions[i][1]
            z = joint[2] * shoulder_lengths[i]
            unnorm_pose.append([x, y, z])
        unnorm_poses.append(unnorm_pose)
    unnorm_poses = np.array(unnorm_poses)
    return unnorm_poses


def rotate_skel(skel_3d, degree):
    """Rotates the skeleton pose

    Keyword Arguments:
    skel_3d - 3D pose, xyz per limb endpoint.
    degree - degrees in radian it needs to be rotated.
    """
    quaternion = Quaternion(axis=[0, 1, 0], angle=math.radians(degree))
    rotated_skel = np.copy(skel_3d)
    n_joints = skel_3d.shape[1]
    for i in range(n_joints):
        rotated_skel[:, i] = quaternion.rotate(skel_3d[:, i])

    return rotated_skel

def augmented_data(skeletons_3d):
    """Creates additional data from 3d skeletons. Rotation + noise"""
    augmented_samples = []
    noise_amount = np.std(skeletons_3d[0][:]) / 10.0
    for skel_3d in skeletons_3d:
        new_sample = rotate_skel(skel_3d, uniform(-20, +20))
        augmented_samples.append(new_sample)
        #adds noise
        noise = np.random.uniform(0, noise_amount,\
            (skel_3d.shape[0], skel_3d.shape[1]))
        new_sample = skel_3d + noise
        new_sample[:, 0] = 0  # neck pos should be zero
        augmented_samples.append(new_sample)
    return augmented_samples

def generate_dataset(raw_dir, dir_name, save_file, data_ratio, valid_joint_index):
    """Generates dataset based on .json frame files from CMU Panoptic dataset
    this method has been updated to work with 18 joints (COCO) instead of 19 (CMU)"""
    # traverse directories
    body_frames_3d = []
    for d_name, subdir_list, files in os.walk(raw_dir):
        if dir_name in d_name:
            print('[INFO] Loading {}'.format(d_name))
            for fname in tqdm(files):
                with open(d_name + '/' + fname) as f:
                    js = json.load(f)
                    for body in js['bodies']:
                        skel_3d = np.array(body['joints19']).reshape((-1, 4))
                        # Ignore poses with less than 50% confidence rate
                        conf = skel_3d[:,3::4]
                        if np.mean(conf) < 0.5:
                            continue
                        # remove dimention of confidence rate 
                        skel_3d = skel_3d[:, :3]
                        body_frames_3d.append(skel_3d[valid_joint_index])
    body_frames_3d = np.array(body_frames_3d)

    print('[INFO] Normalizing...')
    skel_3d, _, _ = normalize_skeleton(body_frames_3d, mode='cmu')
    skel_2d = skel_3d[:, :, :2]
    skel_3d = np.reshape(skel_3d, (-1, len(valid_joint_index)*3))
    skel_2d = np.reshape(skel_2d, (-1, len(valid_joint_index)*2))

    # split train and test data
    print('[INFO] Split Data (Train : Test = {} : 1)'.format(data_ratio))
    data_num = len(skel_2d)
    test_data_num = round(data_num / data_ratio)
    train_data_num = data_num - test_data_num
    train_data = {'src':skel_2d[0:train_data_num], 'tgt':skel_3d[0:train_data_num]}
    test_data = {'src':skel_2d[train_data_num:data_num], 'tgt':skel_3d[train_data_num:data_num]}
    torch.save(train_data, save_file+'_train.pth')
    print('[INFO] Created {}'.format(save_file+'_train.pth'))
    torch.save(test_data, save_file+'_test.pth')
    print('[INFO] Created {}'.format(save_file+'_test.pth'))
    
    print('[INFO] Finish.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('panoptic_data_dir', help='\{DATASET_DIR\} described in the data structure at ReadMe (e.g. "./panoptic_dataset")')
    parser.add_argument('data_suffix', help='suffix of directory name that you downloaded (e.g. hdPose3d_stage1)')
    parser.add_argument('save_dataset_path', help='Path to save the dataset  (e.g. ./data/cmu_dataset)')
    parser.add_argument('--data_ratio', default=10, help='sprit the dataset like Training Data : Test Data = DATA_RATIO : 1')

    args = parser.parse_args()

    VALID_JOINT_INDEX = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

    DATASER_DIR = args.panoptic_data_dir
    SUFFIX = args.data_suffix
    SAVE_FILE_NAME = args.save_dataset_path
    DATA_RATIO = args.data_ratio

    generate_dataset(DATASER_DIR, SUFFIX, SAVE_FILE_NAME, DATA_RATIO, VALID_JOINT_INDEX)
