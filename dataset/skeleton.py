import os
import sys
import numpy as np
import pickle
import torch
import random
from torch.utils.data import DataLoader, Dataset
import copy

from utility.utils import *
# from video_data import *


class Skeleton(Dataset):
    def __init__(self, data_path, label_path, window_size, final_size,
                 split='train', decouple_spatial=0, num_skip_frame=None,
                 random_choose=False, center_choose=False,
                 include_orientation = False):
        """decouple_spatial takes three mode:
            0: no decouple, return data_numpy
            1: return decoupled only
            2: return data_numpy stacked with decoupled at dim=-1 (treat as adding another persion, M+=1)
            3: return data_numpy stacked with decoupled at dim=0 (treat as 6D) 
        """
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.center_choose = center_choose
        self.window_size = window_size
        self.final_size = final_size
        self.num_skip_frame = num_skip_frame
        self.decouple_spatial = decouple_spatial
        self.include_orientation = include_orientation
        self.edge = None
        self.load_data()

    def load_data(self):
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = copy.deepcopy(self.data[index])
        # data_numpy = self.data[index]

        if isinstance(self.label[index], tuple):
            label = (int(self.label[index][0]), int(self.label[index][1]))
        else:
            label = int(self.label[index])
        sample_name = self.sample_name[index]
        data_numpy = np.array(data_numpy)  # C T V M

        data_numpy = data_numpy[:, data_numpy.sum(0).sum(-1).sum(-1) != 0]  # C T' V M (T'<T, padding excluded)

        C, T, V, M = data_numpy.shape

        if T == 0:
            raise ValueError(f'Empty frame detected at index {index}, label: {label}, {sample_name}')
            
        # data transform
        if self.decouple_spatial > 0:
            decoupled = decouple_spatial(data_numpy[:,:,:,0][...,np.newaxis], edges=self.edge)
            if self.decouple_spatial == 1:
                data_numpy = decoupled
            elif self.decouple_spatial == 2:
                data_numpy = np.concatenate((data_numpy, decoupled), axis=-1)
            elif self.decouple_spatial == 3:
                data_numpy = np.concatenate((data_numpy, decoupled), axis=0) 
                
        if self.num_skip_frame is not None:
            velocity = decouple_temporal(data_numpy[:,:,:,0][...,np.newaxis], self.num_skip_frame)
            C, T, V, M = velocity.shape
            data_numpy = np.concatenate((velocity, np.zeros((C, 1, V, M))), 1)
        
        # relative orientation:
        if self.include_orientation:
            orientation = calculate_orientation(data_numpy[:,:,:,0][...,np.newaxis])
            data_numpy = np.concatenate((data_numpy, orientation), axis=-1)

        # data_numpy = pad_recurrent_fix(data_numpy, self.window_size)  # if short: pad recurrent
        # data_numpy = uniform_sample_np(data_numpy, self.window_size)  # if long: resize
        if self.random_choose:
            data_numpy = random_sample_np(data_numpy, self.window_size)
            # data_numpy = random_choose_simple(data_numpy, self.final_size)
        else:
            data_numpy = uniform_sample_np(data_numpy, self.window_size)
        
        if self.center_choose:
            # data_numpy = uniform_sample_np(data_numpy, self.final_size)
            data_numpy = random_choose_simple(data_numpy, self.final_size, center=True)
        else:
            data_numpy = random_choose_simple(data_numpy, self.final_size)

        # if self.split == 'train':
        #     return data_numpy.astype(np.float32), label
        # else:
        #     return data_numpy.astype(np.float32), label, sample_name
        return data_numpy.astype(np.float32), label, sample_name

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def vis(data, edge, is_3d=True, pause=0.01, view=0.25, title='', elev=-90, azim=-90):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    C, T, V, M = data.shape

    plt.ion()
    fig = plt.figure()
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    else:
        ax = fig.add_subplot(111)
    ax.set_title(title)
    p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']

    pose = []
    for m in range(M):
        a = []
        for i in range(len(edge)):
            if is_3d:
                a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
            else:
                a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
        pose.append(a)
    ax.axis([-view, view, -view, view])
    if is_3d:
        ax.set_zlim3d(-view, view)
        ax.view_init(elev=elev, azim=azim)
    for t in range(T):
        for m in range(M):
            for i, (v1, v2) in enumerate(edge):
                pose[m][i].set_xdata(data[0, t, [v1, v2], m])
                pose[m][i].set_ydata(data[1, t, [v1, v2], m])
                if is_3d:
                    pose[m][i].set_3d_properties(data[2, t, [v1, v2], m])

        fig.canvas.draw()
        plt.pause(pause)
    plt.close()
    plt.ioff()

