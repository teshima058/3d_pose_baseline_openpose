import torch
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plot_pose import Plot

def extractSentence(data, mode, index):
    text_id = data[mode]["src"][index]
    texts = []
    sentence = ""
    for id in text_id:
        for word, i in data["dict"].items():
            if id == i:
                texts.append(word)
                sentence += word + ' '
                break
    print('Sentence: ', sentence)
    return sentence

def saveAnimation(pose_data, save_path, fps=12):
    poses = np.array(pose_data)
    p = Plot((-2, 2), (-2, 2))
    anim = p.animate(poses, 1000/fps)
    p.save(anim, save_path, fps=fps)

if __name__ == "__main__":
    base_dir = '../'
    data_path = base_dir + 'data/preprocess_3d.pickle'
    save_path = base_dir + 'videos/'
    mode = 'train'
    index = 1

    data = torch.load(data_path)
    pose_data = data[mode]['tgt']
    poses = pose_data[index]
    # poses = pose_data.to('cpu').detach().numpy()
    poses = data['pca'].inverse_transform(poses)
    print('{} frames'.format(len(poses)))

    extractSentence(data, mode, index)
    saveAnimation(poses, save_path + str(index) + '.mp4', fps=22)