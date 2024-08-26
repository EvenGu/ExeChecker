import argparse
import pickle
import numpy as np
import os, glob
from tqdm import tqdm
import csv


# The longest full sequence in trainset has 897 frames and 768 in valset 
# The longest seq sequence has 104 frames

class1_dict = {'DeepSquat': 0, 
               'HurdleStep': 1, 
               'InlineLunge': 2, 
               'SideLunge': 3, 
               'SitToStand': 4, 
               'StandingASLR': 5, 
               'StandingShoulderABD': 6, 
               'StandingShoulderEXT': 7, 
               'StandingShoulderIRER': 8, 
               'StandingShoulderScaption': 9} # exercise_dict
exe_folders = [f'{exe}{i}' for exe in class1_dict.keys() for i in [1,2]]

class2_dict = {'correct': 1,  'incorrect': 0}

#######################################################################
def ske_vis(data, **kwargs):
    import sys
    sys.path.append('/research/yiwen/ExerciseCheck_HC')
    from dataset.skeleton import vis
    from dataset.uiprmd_skeleton import edge
    vis(data, edge=edge, **kwargs)

def normalize_skeletons(skeleton, origin=0, base_bone=[0,8]):
    '''
    :param skeleton: T, V, C(x, y, z)
    :param origin: int (default: pelvis)
    :param base_bone: [int, int] (default: pelvis-neck)
    :return:
    '''

    T, V, C = skeleton.shape
    if origin is not None:
        # print('sub the center joint #0 (pelvis)')
        main_body_center = skeleton[0, origin].copy()  # c
        skeleton = skeleton - main_body_center 

    if base_bone is not None:
        main_body_spine = np.linalg.norm(skeleton[0, base_bone[1]] - skeleton[0, base_bone[0]])
        skeleton /= main_body_spine

    return skeleton

def read_csv_file(file_path): 
    #TODO: lowerbound (<=), excl (=)
    data_list = []
    # subject001, HurdleStep2.avi, frame_numbers
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        extra = []
        for row in csv_reader:
            if row[0].startswith('subject'):
                frame_numbers = [int(x) for x in row[2:] if x.isnumeric()]
                while frame_numbers[-1] == 0:
                    frame_numbers.pop(-1)
                data_list.append((row[0],row[1], frame_numbers))
            elif row[0] == 'lowerbound' or row[0] == 'excl':
                extra.append(row[:2])
        data_list.append(extra)
    return data_list

#######################################################################################

def gendata(data_root: str, out_path: str, split='val') -> None: 
    val_sub = ['subject010']
    all_sub = [f'subject{i:03d}' for i in range(1,11)]
    if split == 'val':
        sample_set = val_sub
    elif split == 'train':
        sample_set = list(set(all_sub)-set(val_sub))
    else:
        sample_set = all_sub

    sample_set.sort()
    max_frame = 0
    data_list = []
    name_list = [] # f'{sub_id}-{exercise_name}-{correctness}-R{rep_id}': str
    label_list = [] # (exercise_class, correctness_class):(int,int)
    for sub in tqdm(sample_set):
        for exe in exe_folders:
            f = os.path.join(data_root,sub,exe,'pose3D.npy')
            d = np.load(f) # T,V,C
            data_list.append(d)
            if d.shape[0] > max_frame:
                max_frame = d.shape[0]
            correctness = 'correct' if exe[-1]=='1' else 'incorrect'
            name_list.append(f'{sub[-3:]}-{exe[:-1]}-{correctness}')
            label_list.append((class1_dict[exe[:-1]], class2_dict[correctness]))

    print(f'in {split}, num_seq = {len(data_list)}, max_frame = {max_frame}')
    data_skeleton = np.zeros((len(data_list), 3, max_frame, 17, 1), dtype=np.float32)
    for i, data in tqdm(enumerate(data_list)):
        T,V,C = data.shape
        normed_skeletons = normalize_skeletons(data, origin=0, base_bone=[0,8]).transpose((2,0,1)) # T,V,C -> C,T,V
        
        # # for visual check, 
        # skeletons = np.expand_dims(normed_skeletons,-1) # C,T,V -> C,T,V,M
        # ske_vis(skeletons, view=1.5, pause=0.001, elev=15, azim=60, title=f'{name_list[i]}') # z-up

        data_skeleton[i,:,:T,:,0] = normed_skeletons

    with open('{}/{}_label.pkl'.format(out_path, split), 'wb') as f:
        pickle.dump((name_list, label_list), f)
    np.save('{}/{}_data_joint.npy'.format(out_path, split), data_skeleton) # NCTVM

def gendata_seg(data_root: str, seg_file_root: str, out_path: str): # split later

    max_frame_seg = 0    
    data_list = []
    name_list = [] # f'{sub_id}-{exercise_name}-{correctness}-R{rep_id}-{startframe}_{endframe}': str
    label_list = [] # (exercise_class, correctness_class):(int,int)

    for exec in class1_dict.keys(): 
        seg_file = os.path.join(seg_file_root, f'UI-PRMD_SegRep - {exec}.csv')
        seg_list = read_csv_file(seg_file)
        extra = seg_list.pop(-1)
        # print(extra)
        lower_bound = 0
        excl = ''
        for e in extra:
            if e[0] == 'lowerbound':
                lower_bound = int(e[1])
            elif e[0] == 'excl':
                excl = e[1]

        seg_list.sort()
        for sub, exe_vid, frame_numbers in seg_list:

            data_file = os.path.join(data_root, sub, exe_vid[:-4], 'pose3D.npy')
            data = np.load(data_file) # T,V,C
            correctness = 'correct' if exe_vid[-5]=='1' else 'incorrect'
            
            rep_id = 0
            for start_frame, end_frame in zip(frame_numbers[:-1], frame_numbers[1:]):
                if end_frame - start_frame <= lower_bound:
                    # print(f'Skipping segment: {sub}-{exe_vid}-{start_frame:03d}_{end_frame:03d}')
                    continue
                if excl == f'{start_frame}-{end_frame}':
                    # print(f'Skipping segment: {sub}-{exe_vid}-{start_frame:03d}_{end_frame:03d}')
                    continue

                rep_id += 1
                seg_name = f'{sub[-3:]}-{exec}-{correctness}-R{rep_id:02d}'
                # print(seg_name)
                name_list.append(seg_name)
                label_list.append((class1_dict[exec], class2_dict[correctness]))
                d = data[start_frame:end_frame, :, :] # T,V,C
                data_list.append(d)
                if d.shape[0] > max_frame_seg:
                    max_frame_seg = d.shape[0]

    print(f'For segmented sequence: num_seq = {len(data_list)}, max_frame = {max_frame_seg}')
    data_skeleton = np.zeros((len(data_list), 3, max_frame_seg, 17, 1), dtype=np.float32)
    for i, data in tqdm(enumerate(data_list)):
        T,V,C = data.shape
        normed_skeletons = normalize_skeletons(data, origin=0, base_bone=[0,8]).transpose((2,0,1)) # T,V,C -> C,T,V
        
        # # for visual check, 
        # skeletons = np.expand_dims(normed_skeletons,-1) # C,T,V -> C,T,V,M
        # ske_vis(skeletons, view=1.5, pause=0.001, elev=15, azim=60, title=f'{name_list[i]}') # z-up

        data_skeleton[i,:,:T,:,0] = normed_skeletons

    with open('{}/seg_label.pkl'.format(out_path), 'wb') as f:
        pickle.dump((name_list, label_list), f)
    np.save('{}/seg_data_joint.npy'.format(out_path), data_skeleton) # NCTVM


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UI-PRMD Data Converter.')
    parser.add_argument('--data_root', default='/research/yiwen/DATA/UI-PRMD/infer_uiprmd_pf2') 
    parser.add_argument('--seg_file_root', default='/research/yiwen/DATA/UI-PRMD/segmentation_files/') 
    parser.add_argument('--out_path', default='./processed_uiprmd/')

    args = parser.parse_args()
    
    # print(os.getcwd())
    out_path = args.out_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    # for split in ['train', 'val']:
    #     gendata(args.data_root, args.out_path, split=split)

    gendata_seg(args.data_root, args.seg_file_root, args.out_path)