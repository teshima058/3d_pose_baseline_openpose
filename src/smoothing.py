import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def checkOutlier(pose3d, axis='x'):
    upper_joints_idx = [0, 1, 3, 4, 5, 9, 10, 11]
    upper_joints_name = ['neck', 'head', 'R_shoulder', 'R_elbow', 'R_hand', 'L_shoulder', 'L_elbow', 'L_hand']
    x = np.arange(len(pose3d))
    upper_joints = []
    for i in upper_joints_idx:
        upper_joints.append(pose3d[:,i])
    plt.figure()
    if axis  == 'x':
        for i in range(len(upper_joints_idx)):
            plt.plot(x, upper_joints[i][:,0], label=upper_joints_name[i])
    elif axis == 'y':
        for i in range(len(upper_joints_idx)):
            plt.plot(x, upper_joints[i][:,1], label=upper_joints_name[i])
    elif axis == 'z':
        for i in range(len(upper_joints_idx)):
            plt.plot(x, upper_joints[i][:,2], label=upper_joints_name[i])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    plt.close()

def smoothingPose(pose3d, span=20, threshold=2):
    """
    Calculate the mean and standard deviation for each {span}-frame, and linearly interpolate if there are outliers.
    The higher the threshold, the fewer outliers will be detected.
    """

    def zscore(x, axis = None):
        xmean = x.mean(axis=axis, keepdims=True)
        xstd  = np.std(x, axis=axis, keepdims=True)
        zscore = (x-xmean)/xstd
        return zscore

    # ---------- Detect Outliers ----------
    # checkOutlier(pose3d, axis='x')
    data = pose3d.reshape(-1, 19*3)
    outliers = pd.DataFrame([])
    for i,elem in enumerate(data.T):
        df = pd.DataFrame(elem)
        ewm_mean = df.ewm(span=span).mean()
        ewm_std = df.ewm(span=span).std()
        outlier = df[(df - ewm_mean).abs() > ewm_std * threshold]
        outlier.rename(columns={0:i}, inplace=True)
        outliers = pd.concat([outliers, outlier], axis=1)

    # pd.set_option('display.max_rows', None)
    # print(outliers)

    # ---------- Linear Interpolation ----------
    for frame_num in range(outliers.shape[0]):
        for joint_idx in range(outliers.shape[1]):
            if not math.isnan(outliers.iloc[frame_num][joint_idx]):
                if frame_num == 0:
                    data[frame_num][joint_idx] = data[frame_num + 1][joint_idx]
                elif frame_num == outliers.shape[0] - 1:
                    data[frame_num][joint_idx] = data[frame_num - 1][joint_idx]
                else:
                    data[frame_num][joint_idx] = (data[frame_num + 1][joint_idx] + data[frame_num - 1][joint_idx]) / 2
    data = data.reshape([-1, 19, 3])
    # checkOutlier(data, axis='x')

    return data
    
