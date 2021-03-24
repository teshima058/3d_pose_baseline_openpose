import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.plot3D import plot3D, animate3D
from src.poseVisualizer import visualizePose

def CMUPose2KinectData(pose3d, save_mp4=None, save_csv=None):
    """
    pose3d : CMU data ( shape = ({frame_num}, 19, 3) )
    Save the csv file converted to Kinect data for Labanotation
    """

    cmu2kinect = [20, 2, 0, 4, 5, 6, 12, 13, 14, 8, 9, 10, 16, 17, 18]

    kinect_data = np.zeros((len(pose3d), 25, 3))
    for i,pose in enumerate(pose3d):
        # Replacing joints
        for j in range(len(cmu2kinect)):
            kinect_data[i][cmu2kinect[j]] = pose[j]
        # spinMid -> Midpoint between midhip and midshoulder
        kinect_data[i][1] = (pose[0] + pose[2]) / 2
        # head -> Midpoint between both ears
        kinect_data[i][3] = (pose[16] + pose[18]) / 2
        # toes of both feet -> both ankles
        kinect_data[i][15] = pose[8]
        kinect_data[i][19] = pose[14]

        # Rotate x and y axis
        for j in range(len(kinect_data[i])):
            kinect_data[i][j][0] *= -1
            kinect_data[i][j][1] *= -1
        
        # Recoordinate SpineMid to (0, 0, 0)
        spineMid = kinect_data[i][1].copy()
        for j in range(len(kinect_data[i])):
            kinect_data[i][j] -= spineMid

        # Normalized so that the distance between midspine and midhip is 0.25
        spine_hip_length = np.linalg.norm(kinect_data[i][1] - kinect_data[i][0])
        kinect_data[i] = kinect_data[i] * 0.25 / spine_hip_length
    
    # Save MP4
    if save_mp4:
        animate3D(kinect_data, save_path=save_mp4)
        print('Saved mp4 file to {}'.format(save_mp4))

    # Save CSV
    if save_csv:
        # Timestamp for kinect data
        timestamp = np.arange(0, len(pose3d)*50, 50)
        timestamp = timestamp.reshape(-1, 1)
        
        # Extra data for kinect data
        # 0 -> no-detection,    2 -> detected
        kinect_extra = np.zeros((len(pose3d), 25))
        for i in range(len(kinect_extra)):
            for j in range(len(kinect_extra[i])):
                if j in cmu2kinect or j in [1, 3, 15, 19]:
                    kinect_extra[i][j] = 2
                else:
                    kinect_extra[i][j] = 0

        # Concat timestamp and data
        kinect_extra = kinect_extra.reshape(-1, 25, 1)
        tmp = np.concatenate([kinect_data, kinect_extra], axis=2)
        tmp = tmp.reshape(-1, 25 * 4)
        kinect_csv_data = np.concatenate([timestamp, tmp], axis=1)


        # Save
        kinect_csv_data = pd.DataFrame(kinect_csv_data)
        for i in range(0, 101, 4):
            kinect_csv_data[i] = kinect_csv_data[i].astype('int')
        kinect_csv_data.to_csv(save_csv, header=False, index=False)

    