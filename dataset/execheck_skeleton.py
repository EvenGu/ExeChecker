
import pickle
from torch.utils.data import DataLoader
import numpy as np
import os
from dataset.skeleton import Skeleton, vis

import random
from joints_dtw import exercise_names, JOINTS

edge17 = [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (2, 7), (7, 8), (8, 9), 
          (0, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (3, 16)] # proximal->distal

edge21 = edge17 + [(6, 17), (9, 18), (12, 19), (15, 20)] # more edges

hierarchy = {i:[i,i+10] for i in range(10)} # {exe:[ie,ce]}


class EXECHECK_SKE(Skeleton):
    def __init__(self, folder_path, window_size, final_size, split='train', decouple_spatial=False,
                 num_skip_frame=None, random_choose=False, center_choose=False, 
                 num_joints = 17,
                 use_subclass=False, use_hierachy=False, 
                 include_orientation=False):
        self.use_subclass = use_subclass # this is Not in Skeleton
        self.use_hierachy = use_hierachy # TODO
        data_path = os.path.join(folder_path, f'seg_data_joint_{split}.npy')
        label_path = os.path.join(folder_path, f'seg_label_{split}.pkl')
        self.num_joints = num_joints
        super().__init__(data_path, label_path, window_size, final_size, split, decouple_spatial, num_skip_frame,
                         random_choose, center_choose, include_orientation)
        
        if self.num_joints == 17: # no hand nor foot
            self.edge = tuple(edge17) # edge is init as None in Skeleton
        elif self.num_joints == 21:
            self.edge = tuple(edge21)

    def load_data(self):
        with open(self.label_path, 'rb') as f:
            self.sample_name, labels = pickle.load(f)
        if self.use_hierachy: # each label is a tuple
            self.label = [(lb[0], (lb[1])*10+(lb[0])) for lb in labels] 
            # print(self.label)
        else:
            if self.use_subclass:
                # each label includes [type_class, correctness, (SLR)]
                # correct: 1, incorrect:0 
                #   --> ce in [10-19], ie in [0-9]
                sublabel = [(lb[1])*10+(lb[0]) for lb in labels]
                # assert set(sublabel) == set(range(20))
                self.label = sublabel
            else: 
                self.label = [lb[0] for lb in labels] # zero-index (0-9)
        
        self.data = np.load(self.data_path, mmap_mode='r')  # NCTVM
        self.data = self.data[:,:,:,:self.num_joints,:]
        # print(f"EXECHECK SKE load_data, shape = {self.data.shape}") 





def test(folder_path, vid=None, edge=None, is_3d=False, split='train'):
    dataset = EXECHECK_SKE(folder_path, window_size=160, final_size=128, split=split,
                      random_choose=True, center_choose=False, use_subclass=True)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    # if split=='train':
    #     labels = open('../prepare/execheck/label.txt', 'r').readlines()
    #     labels.sort()
    #     for i, (data, label) in enumerate(loader):
    #         if i%10 ==0:
    #             print(i, data.shape, label.item())
    #             vis(data[0].numpy(), edge=edge, view=1, pause=0.1, title=labels[label.item()-1].rstrip())
    # else: 
    #     for i, (data, label, name) in enumerate(loader):
    #         if i%2 !=0:
    #             print(i, data.shape, label.item())
    #             vis(data[0].numpy(), edge=edge, view=1, pause=0.1, title=name)

    sample_name = loader.dataset.sample_name
    print(sample_name)
    # sample_id = [name.split('-')[1] for name in sample_name]
    index = sample_name.index(vid)
    if split == 'train':
        data, label = loader.dataset[index]
    else:
        data, label, name = loader.dataset[index]
    # skeleton
    vis(data, edge=edge, view=1, pause=0.01, title=vid)


def save_sample_json (data, out_path):
    C, T, V = data.shape
    mydict = {i:{} for i in range(T)}
    for i in range(T):
        for kp_i,kp in enumerate(JOINTS):
            mydict[i][kp] = data[:,i,kp_i].tolist()
    import json
    json.dump(mydict,open(out_path,'w'))
    

if __name__ == '__main__':

    split = 'val'
    folder_path = f"../processed_execheck/xsub/"

    test(folder_path, vid='6-forward_lunge-incorrect-R1', edge=edge21, is_3d=True, split='val')

    # dataset = EXECHECK_SKE(folder_path, window_size=160, final_size=128, split=split,
    #                        decouple_spatial=True, include_orientation=True, use_hierachy=True)
    # loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    # sample_names = loader.dataset.sample_name

    # subs = ['m6', '6']
    # reps = [f'R{i}' for i in range(1,6)]
    # for exe in exercise_names:
    #     for form in ['correct', 'incorrect']:
    #         s = random.choice(subs)
    #         rep = random.choice(reps)
    #         name = f'{s}-{exe}-{form}-{rep}'
    #         index = sample_names.index(name)
    #         data, label, _ = loader.dataset[index] # CTVM (T=final_size)
    #         data = data.squeeze() # CTV
    #         # print(f'{data.shape}, {label}') 
    #         np.save(f'../data_samples/sample_npy/{exe}-{form}.npy', data)
    #         save_sample_json(data, f'../data_samples/sample_json/{exe}-{form}.json')
    

    # item = loader.dataset[7]
    # print(item[0].shape) # NCVTM, M=3
    # print(item[1]) # numeric_label
    # if split == 'val':
    #     print(item(2))  # name