import os
import sys
import re
import json
import numpy as np
import argparse 
from tqdm import tqdm

import torch
from torch.autograd import Variable

from generate_dataset import normalize_skeleton, unNormalize_skeleton
from src.model import LinearModel, weight_init
from src.smoothing import smoothingPose
from src.cmu2kinect import CMUPose2KinectData
from src.plot3D import plot3D, animate3D
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

coco_joint_index = [''] * 18
coco_joint_index[0]  = 'Nose'
coco_joint_index[1]  = 'Neck'
coco_joint_index[2]  = 'RShoulder'
coco_joint_index[3]  = 'RElbow'
coco_joint_index[4]  = 'RWrist'
coco_joint_index[5]  = 'LShoulder'
coco_joint_index[6]  = 'LElbow'
coco_joint_index[7]  = 'LWrist'
coco_joint_index[8]  = 'RHip'
coco_joint_index[9]  = 'RKnee'
coco_joint_index[10] = 'RAnkle'
coco_joint_index[11] = 'LHip'
coco_joint_index[12] = 'LKnee'
coco_joint_index[13] = 'LAnkle'
coco_joint_index[14] = 'REye'
coco_joint_index[15] = 'LEye'
coco_joint_index[16] = 'REar'
coco_joint_index[17] = 'LEar'

coco2cmu = [1, 0, 9, 10, 11, 3, 4, 5, 12, 13, 14, 6, 7, 8, 17, 15, 18, 16]

class PoseBaselineForCOCO():
    def __init__(self, pretrained_model_path):
        ckpt = torch.load(pretrained_model_path)
        self.mean_pose = ckpt['mean_pose']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = LinearModel(joint_num=19)
        self.model.cuda()
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

    def read_openpose_json(self, openpose_output_dir):
        # load json files
        json_files = os.listdir(openpose_output_dir)
        # check for other file types
        json_files = sorted([filename for filename in json_files if filename.endswith(".json")])

        pose2d, confs = [], []
        ### extract x,y and ignore confidence rate
        for file_name in json_files:
            _file = os.path.join(openpose_output_dir, file_name)
            data = json.load(open(_file))
            if len(data['people']) == 0:
                continue

            # get frame INDEX from 12 digit number string
            frame_indx = re.findall("(\d{12})", file_name)
            # if int(frame_indx[0]) <= 0:
            # n order to register the first frame as it is, specify INDEX as it is
            tmp = data["people"][0]["pose_keypoints_2d"]

            # if openpose is 25 joints version -> convert to 18 joints
            if len(tmp) == 75:
                # remove joint index
                remove_idx = [8, 19, 20, 21, 22, 23, 24]
                tmp = np.array(tmp).reshape([-1,3])
                tmp = np.delete(tmp, remove_idx, axis=0)
                tmp = tmp.reshape(-1)

            pose_frame, conf = [], []
            for i in range(len(tmp)):
                if i % 3 == 2:
                    conf.append(tmp[i])
                else:
                    pose_frame.append(tmp[i])
            confs.append(conf)
            pose2d.append(pose_frame)

        return np.array(pose2d), np.array(confs)
    
    # COCO-Data(18 Joints) -> CMU-Data(19 Joints)
    def convertCOCO2CMU(self, pose2d, conf_rate):
        pose2d = pose2d.reshape(-1, 18, 2)
        cmu_poses = []
        cmu_confs = []
        for i,pose in enumerate(pose2d):
            cmu_pose = [None] * 19
            cmu_conf = [None] * 19
            for j in range(len(pose)):
                cmu_pose[coco2cmu[j]] = pose[j]
                cmu_conf[coco2cmu[j]] = conf_rate[i][j]
            cmu_pose[2] = (pose[8] + pose[11]) / 2                      # MidHip
            cmu_conf[2] = (conf_rate[i][8] + conf_rate[i][11]) / 2      # MidHip
            cmu_poses.append(cmu_pose)
            cmu_confs.append(cmu_conf)
        cmu_poses = np.array(cmu_poses)
        cmu_confs = np.array(cmu_confs)
        return cmu_poses, cmu_confs
    
    # linealy interpolate joints that not estimated by OpenPose 
    def linearInterpolation(self, skeletons, conf_rate):
        conf_rate = conf_rate.T
        skeletons = skeletons.reshape([-1, skeletons.shape[1]*skeletons.shape[2]]).T
        
        # At first, if confidence rate of first or end frame is 0, it is interpolated with the nearest value
        for joint_idx in range(len(conf_rate)):
            # First frame
            if conf_rate[joint_idx][0] == 0:
                for i in range(1, len(conf_rate[joint_idx])):
                    if conf_rate[joint_idx][i] != 0:
                        skeletons[joint_idx*2+0][0] = skeletons[joint_idx*2+0][i]
                        skeletons[joint_idx*2+1][0] = skeletons[joint_idx*2+1][i]
            # End frame
            end_frame = len(conf_rate[joint_idx])-1
            if conf_rate[joint_idx][end_frame] == 0:
                for i in range(end_frame-1, 0, -1):
                    if conf_rate[joint_idx][i] != 0:
                        skeletons[joint_idx*2+0][end_frame] = skeletons[joint_idx*2+0][i]
                        skeletons[joint_idx*2+1][end_frame] = skeletons[joint_idx*2+1][i]
        
        # Second detect outliers
        outliers = []       # frames to interpolate for each joint
        for joint_idx in range(len(conf_rate)):
            outlier = []
            i = 0
            for frame in range(len(conf_rate[joint_idx])):
                if frame < i or frame == 0 or frame == len(conf_rate[joint_idx])-1:
                    continue
                if conf_rate[joint_idx][frame] == 0:
                    out = []
                    i = frame
                    skip = False
                    while(conf_rate[joint_idx][i] == 0):
                        out.append(i)
                        i += 1
                        if i > len(conf_rate[joint_idx]) - 1:
                            skip = True
                            break
                    if not skip:
                        outlier.append([out[0], out[len(out)-1]])
            outliers.append(outlier)

        # Finally Linear Interpolation
        for joint in range(len(outliers)):
            for frame in outliers[joint]:
                j = 1
                for i in range(frame[0], frame[1] + 1):
                    # Interpolation
                    skeletons[joint*2+0][i] = skeletons[joint*2+0][frame[0]-1] + j * (skeletons[joint*2+0][frame[1]+1] - skeletons[joint*2+0][frame[0]-1]) / (frame[1] - frame[0] + 2)
                    skeletons[joint*2+1][i] = skeletons[joint*2+1][frame[0]-1] + j * (skeletons[joint*2+1][frame[1]+1] - skeletons[joint*2+1][frame[0]-1]) / (frame[1] - frame[0] + 2)
                    j += 1
        skeletons = skeletons.T.reshape([-1, 19, 2])
        return skeletons

    # inputs : torch_tensor[batch_size][joints(19*2)]
    # For data that include only upper body, fill the lower body with mean_pose.
    def predict(self, openpose_json_dir):
        pose2d, confs = self.read_openpose_json(openpose_json_dir)

        # Check if only the upper body is detected
        lower_body_conf = np.array(confs)[:,8:14]
        if np.mean(lower_body_conf) < 0.4:
            isUpperBody = True
        else:
            isUpperBody = False

        # convert COCO joints index to CMU joints index
        pose2d, confs = self.convertCOCO2CMU(pose2d, confs)
        
        # Normalize input pose
        inputs, shoulder_len, neck_pos = normalize_skeleton(pose2d, mode='cmu')

        # Linear interpolation of unestimated joints
        inputs = self.linearInterpolation(inputs, confs)
        
        # if only upperbody is estimated, fill the lower body with mean pose
        if isUpperBody:
            upper_joints = [0, 1, 3, 4, 5, 9, 10, 11]
            for i in range(len(inputs)):
                for j in range(len(inputs[i])):
                    if not j in upper_joints:
                        inputs[i][j] = self.mean_pose[j][0:2]

        inputs = Variable(torch.tensor(inputs).cuda().type(torch.cuda.FloatTensor)).reshape(-1, 38)
        outputs = self.model(inputs)
        outputs = outputs.cpu().detach().numpy().reshape(-1, 19, 3)
        # outputs = unNormalize_skeleton(outputs, shoulder_len, neck_pos, mode='cmu')
        return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('pretrain_model_path', help='Pretrained model path (e.g. ./checkpoint/best.chkpt)')
    parser.add_argument('openpose_json_dir', help='Directory containing the json files output by OpenPose')
    parser.add_argument('--save_mp4', help='Path to save the video (e.g. ./videos/output.mp4)')
    parser.add_argument('--save_csv', help='Path to save the csv file (e.g. ./output/output.csv)')

    args = parser.parse_args()


    # ----- Predict -----
    print('Predicting 3D-Pose from {}'.format(args.openpose_json_dir))
    p = PoseBaselineForCOCO(args.pretrain_model_path)
    pose3d = p.predict(args.openpose_json_dir)
    pose3d = smoothingPose(pose3d)


    # ----- Plot sample pose -----
    # visualizePose(pose3d[0], mode='cmu')
    # visualizePose(pose3d[10], mode='cmu')
    # visualizePose(pose3d[20], mode='cmu')


    # ----- Save animation ------
    if args.save_mp4:
        animate3D(pose3d, save_path=args.save_mp4)
        # animate3D(pose3d, save_path="./videos/upperbody_smoothed.mp4", isUpperBody=True)


    # ----- Save kinect format csv file -----
    if args.save_csv:
        CMUPose2KinectData(pose3d, save_csv=args.save_csv)

    print('Finish')