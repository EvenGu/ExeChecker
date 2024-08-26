# import torch.utils.data as data
# from PIL import Image

# import errno
import torch
from torch.utils.data import Dataset
# import json
# import codecs
import csv
import os
import pickle as pkl
import itertools

from dataset.execheck_skeleton import EXECHECK_SKE
# import sys
# sys.path.extend(['../'])
from utility.get_triplets_list import read_triplets_list
        
        
class TripletDataset(Dataset):
    def __init__(self, exec_dataset, split, exec_class, folder_path):
        assert type(exec_class) == list, 'exercise_class should be a list a list of integers'
        self.exec_dataset = exec_dataset
        self.split = split
        self.exec_class = exec_class
        self.triplets_idx = self.load_triplets(folder_path)

    def load_triplets(self, folder_path):
        triplets_idx = []  # indices (a,p,n) in exec_dataset
        for e_cls in self.exec_class: 
            triplet_file = f'class-{e_cls}_triplet_{self.split}.txt'
            triplets_idx.extend(read_triplets_list(os.path.join(folder_path,'triplets',triplet_file)))

        return triplets_idx


    def __len__(self):
        return len(self.triplets_idx)

    def __getitem__(self, index):
        anchor_idx, positive_idx, negative_idx = self.triplets_idx[index]
        # TODO: do i need to include labels?
        #### returned by exec_dataset[idx] ####
        # if self.split == 'train':
        #     return data_numpy.astype(np.float32), label
        # else:
        #     return data_numpy.astype(np.float32), label, sample_name

        anchor = self.exec_dataset[anchor_idx]
        positive = self.exec_dataset[positive_idx]
        negative = self.exec_dataset[negative_idx]

        # # sanity check
        # for ele in [anchor, positive, negative]:
        #     chosen_class = [cls+10 for cls in self.exec_class] + self.exec_class
        #     # print(chosen_class)
        #     if ele[1] not in chosen_class:
        #         print('something wrong')
        #         print('index in triplet loader: ', index)
        #         print('..corresponds to ', self.triplets_idx[index])
        #         print(ele[1], ele[2])

        return anchor, positive, negative

        

if __name__ == '__main__':

    from torch.utils.data import DataLoader
    folder_path = '../processed_execheck/xsub'
    split = 'val'
    data_path = os.path.join(folder_path, f'seg_data_joint_{split}.npy')
    label_path = os.path.join(folder_path, f'seg_label_{split}.pkl')
    exec_dataset = EXECHECK_SKE(folder_path, window_size=60, final_size=50, split=split, use_subclass=True)

    # Initialize the TripletDataset
    triplet_dataset = TripletDataset(exec_dataset, split, exec_class=[0,9,4],   
                                     folder_path=folder_path)

    # Create a DataLoader for the triplet dataset
    triplet_loader = DataLoader(triplet_dataset, batch_size=5, shuffle=True)

    for batch_idx, (anchors, positives, negatives) in enumerate(triplet_loader):

        # if batchsize==1
        # for ele in (anchors, positives, negatives):
            # [print(name, '\t\t', lbl.item()) for name, lbl in zip(ele[2], ele[1]) ]

        print('anchors, positives, negatives')
        # print((anchors[1].item(), positives[1].item(), negatives[1].item())) 
        print((anchors[2], positives[2], negatives[2]))

        
            
