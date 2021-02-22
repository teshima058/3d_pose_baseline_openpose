import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d, Axes3D

def plot3D(pose, lim=[-3, 3], gt=None, save_path=""):
    fig = plt.figure(figsize=(10, 10))
    ax_3d = fig.add_subplot(1, 1, 1, projection='3d')

    edges = np.array([[0, 1], [1, 2],[2, 3], [3, 4], [1, 5], [5, 6], [6, 7]])

    for edge in edges:
        x = np.array([pose[edge[0]*3],   pose[edge[1]*3]])
        y = np.array([pose[edge[0]*3+1], pose[edge[1]*3+1]])
        z = np.array([pose[edge[0]*3+2], pose[edge[1]*3+2]])
        ax_3d.plot(x, z, y, lw=2, color='r')
        if gt is not None:
            x_gt = np.array([gt[edge[0]*3],   gt[edge[1]*3]])
            y_gt = np.array([gt[edge[0]*3+1], gt[edge[1]*3+1]])
            z_gt = np.array([gt[edge[0]*3+2], gt[edge[1]*3+2]])
            ax_3d.plot(x_gt, z_gt, y_gt, lw=2, color='g')

    # ax_3d.set_aspect('equal')
    ax_3d.set_xlabel("x"), ax_3d.set_ylabel("z"), ax_3d.set_zlabel("y")
    ax_3d.set_xlim3d(lim), ax_3d.set_ylim3d([lim[1],lim[0]]), ax_3d.set_zlim3d(lim)
    # ax_3d.view_init(azim=-90, elev=90)    # above
    ax_3d.view_init(azim=-90, elev=0)     # front
    # ax_3d.view_init(azim=0, elev=15)      # side

    if save_path:
        # plt.title(save_path[-6:-4])
        plt.savefig(save_path)
    else:
        plt.show()

def animate3D(poses, save_path, fps=25, fig_scale=1.5, rotation=40, gt=None, isUpperBody=False):
    fig = plt.figure(figsize=(10, 10))
    # ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
    ax_3d = fig.gca(projection='3d')
    if isUpperBody:
        edges = np.array([[0, 1], [0, 3], [3, 4], [4, 5], [0, 9], [9, 10], [10, 11]])
    else:
        edges = np.array([[0,1], [0,2], [0,3], [3,4], [4,5], [2,6], [6,7], [7,8], [0,9], [9,10], [10,11], [2,12], [12,13], [13,14], [1,15], [15,16], [1,17], [17,18]])

    nan_index = []
    for i,isNanArray in enumerate(np.isnan(poses).any(axis=1)):
        if True in isNanArray:
            nan_index.append(i)
    poses = np.delete(poses, nan_index, axis=0)

    # ax_3d.set_aspect('equal')
    ax_3d.set_xlabel("x"), ax_3d.set_ylabel("z"), ax_3d.set_zlabel("y")
    mean = np.mean(np.mean(poses, axis=0), axis=0)
    shoulder_len = np.mean(poses[:,3,:][:,0]) - np.mean(poses[:,9,:][:,0])
    ax_3d.set_xlim3d([mean[0] - shoulder_len * fig_scale, mean[0] + shoulder_len * fig_scale])
    if isUpperBody:
        ax_3d.set_zlim3d([mean[1] - shoulder_len * fig_scale, mean[1] + shoulder_len * fig_scale])
    else:
        ax_3d.set_zlim3d([mean[1] - shoulder_len * fig_scale * 2 + shoulder_len * 1, mean[1] + shoulder_len * fig_scale * 2 + shoulder_len * 1])
    ax_3d.set_ylim3d([mean[2] - shoulder_len * fig_scale, mean[2] + shoulder_len * fig_scale])
    ax_3d.view_init(azim=rotation, elev=195)
    
    p = []
    p_gt = []
    for i in range(len(edges)):
        p.append(ax_3d.plot([], [], [], lw=2, color='r'))
        p_gt.append(ax_3d.plot([], [], [], lw=2, color='g'))

    def update(i):
        for j,edge in enumerate(edges):
            x = np.array([poses[i][edge[0]][0], poses[i][edge[1]][0]])
            y = np.array([poses[i][edge[0]][1], poses[i][edge[1]][1]])
            z = np.array([poses[i][edge[0]][2], poses[i][edge[1]][2]])
            p[j][0].set_data((x, z))
            p[j][0].set_3d_properties(y)
        # return p[0][0], p[1][0], p[2][0], p[3][0], p[4][0], p[5][0], p[6][0]

    def update_gt(i):
        for j,edge in enumerate(edges):
            x = np.array([poses[i][edge[0]][0], poses[i][edge[1]][0]])
            y = np.array([poses[i][edge[0]][1], poses[i][edge[1]][1]])
            z = np.array([poses[i][edge[0]][2], poses[i][edge[1]][2]])
            x_gt = np.array([gt[i][edge[0]][0],   gt[i][edge[1]][0]])
            y_gt = np.array([gt[i][edge[0]][1], gt[i][edge[1]][1]])
            z_gt = np.array([gt[i][edge[0]][2], gt[i][edge[1]][2]])
            p[j][0].set_data((x, z))
            p[j][0].set_3d_properties(y)
            p_gt[j][0].set_data((x_gt, z_gt))
            p_gt[j][0].set_3d_properties(y_gt)
        return p[0][0], p[1][0], p[2][0], p[3][0], p[4][0], p[5][0], p[6][0], \
                p_gt[0][0], p_gt[1][0], p_gt[2][0], p_gt[3][0], p_gt[4][0], p_gt[5][0], p_gt[6][0]

    if gt is not None:
        ani = animation.FuncAnimation(fig, update_gt, len(poses), interval=1000/fps)
    else:
        ani = animation.FuncAnimation(fig, update, len(poses), interval=1000/fps)
    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(save_path, writer=writer)
    print("Saved mp4 file to ", save_path)

def main():
    data_path = './data/cmu_dataset_tmp_train.pickle'
    data = torch.load(data_path)
    pose3d = data['tgt']

    plot3D(pose3d[0][0])

    # ---- save 3D animation file -----
    direction = 'above'
    save_path = './videos/test_gt.mp4'
    animate3D(pose3d[0][300:500], gt=None, direction=direction, fps=10, lim=[-2,2], save_path=save_path)

if __name__ == "__main__":
    main()