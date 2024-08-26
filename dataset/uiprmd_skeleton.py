import pickle
from utility.video_data import *
from dataset.execheck_skeleton import EXECHECK_SKE
import os

edge = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
        [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

bone_LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool) # 0:R, 1:L

class UIPRMD_SKE(EXECHECK_SKE): # PFV2 format
    def __init__(self, folder_path, window_size, final_size, split='train', decouple_spatial=False,
                 num_skip_frame=None, random_choose=False, center_choose=False, 
                 num_joints = 17,
                 use_subclass=False, use_hierachy=False, include_orientation=False):
        
        super().__init__(folder_path, window_size, final_size, split, decouple_spatial, num_skip_frame,
                         random_choose, center_choose, 
                         num_joints, use_subclass, use_hierachy, include_orientation)
        
        self.edge = edge
    