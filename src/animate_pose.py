import torch
import numpy as np
import matplotlib.pyplot as plt
from src.plotPose import Plot


def plotUpperBody(pose_data, save_path, fps=12):
    upper_idx = [1, 0, 3, 4, 5, 9, 10, 11, 15, 16, 17, 18]
    pose2d = pose_data[:, upper_idx]
    # rotate y axis
    for frame in range(len(pose2d)):
        for joint in range(len(pose2d[frame])):
            pose2d[frame][joint][1] *= -1
    pose2d = pose2d.reshape([-1, len(upper_idx)*2])
    time = np.arange(len(pose2d))
    time = time.reshape(-1, 1)
    pose2d = np.concatenate([pose2d, time], axis=1)
    p = Plot((0, 1280), (-720, 0))
    anim = p.animate(pose2d, 1000/fps)
    p.save(anim, save_path, fps=fps)

