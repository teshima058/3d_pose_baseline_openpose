import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# pose.shape : ({joint_num}, 3)
# index_list : np.arange({joint_num})
def visualizePose(pose, mode=None, fig_scale=1, index_list=None,):
    fig = plt.figure()
    ax = fig.add_subplot(111 , projection='3d')
    for i,p in enumerate(pose):
        if mode == 'upper':
            upper_joints = [0, 1, 3, 4, 5, 9, 10, 11]
            if not i in upper_joints:
                continue
            ax.scatter(p[0], p[1], zs=p[2], zdir='z', s=20, marker='o', cmap=plt.cm.jet) 
        elif index_list is not None:
            ax.scatter(p[0], p[1], zs=p[2], zdir='z', s=50, marker='${}$'.format(index_list[i]), cmap=plt.cm.jet) 
        else:
            ax.scatter(p[0], p[1], zs=p[2], zdir='z', s=20, marker='o', cmap=plt.cm.jet) 

    if mode == 'h36':
        borne_list = [[0,1], [0,4], [1,2], [2,3], [4,5], [5,6], [0,7], [7,8], [8,9], [9,10], [8,11], [11,12], [12,13], [8,14], [14,15], [15,16]]
        for b in borne_list:
            ax.plot([pose[b[0]][0], pose[b[1]][0]], [pose[b[0]][1], pose[b[1]][1]], [pose[b[0]][2], pose[b[1]][2]], lw=2)
    elif mode == 'cmu':
        borne_list = [[0,1], [0,2], [0,3], [3,4], [4,5], [2,6], [6,7], [7,8], [0,9], [9,10], [10,11], [2,12], [12,13], [13,14], [1,15], [15,16], [1,17], [17,18]]
        for b in borne_list:
            ax.plot([pose[b[0]][0], pose[b[1]][0]], [pose[b[0]][1], pose[b[1]][1]], [pose[b[0]][2], pose[b[1]][2]], lw=2)
    elif mode == 'upper':
        borne_list = [[0, 1], [0, 3], [3, 4], [4, 5], [0, 9], [9, 10], [10, 11]]
        for b in borne_list:
            ax.plot([pose[b[0]][0], pose[b[1]][0]], [pose[b[0]][1], pose[b[1]][1]], [pose[b[0]][2], pose[b[1]][2]], lw=2)

    # ax.set_xlim(-3 * fig_scale, 3 * fig_scale)
    # ax.set_ylim(-4 * fig_scale, 8 * fig_scale)
    # ax.set_zlim(-3 * fig_scale, 3 * fig_scale)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.show()


# Example
# --------------------------
# visualize outputs
# --------------------------
# ~~~
# outputs = model(inputs)
# 
# idx = 0
# pose = outputs[idx].reshape(-1, 3)
# pose = pose.cpu().detach().numpy()
# index_frame = np.arange(len(pose))
# visualizePose(pose, index_frame)



# --------------------------
# visualize inputs
# --------------------------
# idx = 0
# pose = []
# for i in range(len(inputs[idx])):
#     pose.append(inputs[idx][i].cpu().detach().numpy())
#     if i % 2 == 1:
#         pose.append(0)
# pose = np.array(pose)
# pose = np.reshape(pose, [-1, 3])
# index_frame = np.arange(len(pose))
# visualizePose(pose, index_frame)