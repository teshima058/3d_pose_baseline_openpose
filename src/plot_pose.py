import matplotlib.pyplot as plt
import numpy as np
import argparse
import seaborn as sns
import pandas as pd

from matplotlib import pyplot, transforms
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.animation import FFMpegWriter
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

class Plot():

    def __init__(self, x_lim, y_lim):
        """Instantiates an object to visualize the generated poses."""
        self.fig = plt.figure()
        ax = plt.axes(xlim=x_lim, ylim=y_lim)
        
        [self.line] = ax.plot([], [], lw=5)
        
        [self.line2] = ax.plot([], [], lw=5)
        [self.line3] = ax.plot([], [], lw=5)
        [self.line4] = ax.plot([], [], lw=5)

        [self.line5] = ax.plot([], [], lw=5)
        [self.line6] = ax.plot([], [], lw=5)
        [self.line7] = ax.plot([], [], lw=5)

    def init_line(self):
        """Creates line objects which are drawn later."""
        self.line.set_data([], [])
        
        self.line2.set_data([], [])
        self.line3.set_data([], [])
        self.line4.set_data([], [])

        self.line5.set_data([], [])
        self.line6.set_data([], [])
        self.line7.set_data([], [])
        
        return ([self.line, self.line2, self.line3, self.line4, self.line5, self.line6, self.line7])

    def animate_frame(self, pose):
        if len(pose) == 24:
            self.line.set_data(pose[0:6:3], pose[1:6:3])
            self.line2.set_data(pose[3:9:3], pose[4:9:3])
            self.line3.set_data([pose[6], pose[9]], [pose[7], pose[10]])
            self.line4.set_data([pose[12], pose[9]], [pose[13], pose[10]])
            self.line5.set_data([pose[3],pose[15]], [pose[4],pose[16]])
            self.line6.set_data([pose[15], pose[18]], [pose[16], pose[19]])
            self.line7.set_data([pose[21], pose[18]], [pose[22], pose[19]])
        elif len(pose) == 16:
            self.line.set_data([pose[0], pose[2]], [pose[1], pose[3]])
            self.line2.set_data([pose[2], pose[4]], [pose[3], pose[5]])
            self.line3.set_data([pose[4], pose[6]], [pose[5], pose[7]])
            self.line4.set_data([pose[6], pose[8]], [pose[7], pose[9]])
            self.line5.set_data([pose[2],  pose[10]], [pose[3],  pose[11]])
            self.line6.set_data([pose[10], pose[12]], [pose[11], pose[13]])
            self.line7.set_data([pose[12], pose[14]], [pose[13], pose[15]])
        return ([self.line, self.line2, self.line3, self.line4, self.line5, self.line6, self.line7])

    def animate(self, frames_to_play, interval):
        """Returns a matplotlib animation object that can be saved as a video."""
        anim = animation.FuncAnimation(self.fig, self.animate_frame, 
                                        init_func=self.init_line, frames=frames_to_play, 
                                        interval=interval, blit=True)

        return anim

    def save(self, ani, name, fps=10):
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('{}'.format(name), writer=writer)
        print("[INFO] {} file saved.".format(name))


def display_pose(pose, degree=0, linewidth=5.0):
    base = pyplot.gca().transData
    rot = transforms.Affine2D().rotate_deg(degree)

    if len(pose) == 24:
        plt.plot(pose[0:6:3], pose[1:6:3], transform=rot+base, linewidth=linewidth) #neck
        plt.plot(pose[3:9:3], pose[4:9:3], transform=rot+base,linewidth=linewidth) #sholder 1
        plt.plot([pose[6], pose[9]], [pose[7], pose[10]], transform=rot+base, linewidth=linewidth)
        plt.plot([pose[12], pose[9]], [pose[13], pose[10]], transform=rot+base, linewidth=linewidth) #arm1-1
        plt.plot([pose[3],pose[15]], [pose[4],pose[16]], transform=rot+base, linewidth=linewidth) #sholder 2
        plt.plot([pose[15], pose[18]], [pose[16], pose[19]], transform=rot+base, linewidth=linewidth)
        plt.plot([pose[21], pose[18]], [pose[22], pose[19]], transform=rot+base, linewidth=linewidth) #arm2-1
    elif len(pose) == 16:
        plt.plot([pose[0], pose[2]], [pose[1], pose[3]], transform=rot+base, linewidth=linewidth) #neck
        plt.plot([pose[2], pose[4]], [pose[3], pose[5]], transform=rot+base,linewidth=linewidth) #sholder 1
        plt.plot([pose[4], pose[6]], [pose[5], pose[7]], transform=rot+base, linewidth=linewidth)
        plt.plot([pose[6], pose[8]], [pose[7], pose[9]], transform=rot+base, linewidth=linewidth) #arm1-1
        plt.plot([pose[2],  pose[10]], [pose[3],  pose[11]], transform=rot+base, linewidth=linewidth) #sholder 2
        plt.plot([pose[10], pose[12]], [pose[11], pose[13]], transform=rot+base, linewidth=linewidth)
        plt.plot([pose[12], pose[14]], [pose[13], pose[15]], transform=rot+base, linewidth=linewidth) #arm2-1
    else:
        print('[ERROR] Wrong Pose Format')
        exit(-1)


def display_multi_poses(poses, col=10):
    row = poses.shape[0] / col
    fig = plt.figure(dpi=100, figsize=(13.4, 5.4))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    for i in range(1, poses.shape[0]+1):
        plt.subplot(row, col, i)
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.xticks(color='None')
        plt.yticks(color='None')
        plt.tick_params(length=0)
        display_pose(poses[i-1], linewidth=3.0)


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-model', default='seq2pos')
    parser.add_argument('-model', default='transformer')
    parser.add_argument('-log', default='./log/')
    opt = parser.parse_args()

    display_loss(opt.log+opt.model+'_train.log', opt.log+opt.model+'_valid.log')
    plt.savefig(opt.log+'loss.png')


if __name__=='__main__':
    main()