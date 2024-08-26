import argparse
import pickle, json
import numpy as np
import os, glob
from tqdm import tqdm
import sys
sys.path.extend(['../../', '../'])

INCL = [0, 1, 2, 3, 5, 6, 7, 12, 13, 14, 18, 19, 20, 22, 23, 24, 27, 
        8, 15, 21, 25] # left right hand foot
num_joints = len(INCL)

exercise_names = ['arm_circle', 
                  'forward_lunge', 
                  'high_knee_raise', 
                  'hip_abduction', 
                  'leg_extension', 
                  'shoulder_abduction', 
                  'shoulder_external_rotation', 
                  'shoulder_flexion', 
                  'side_step_squat', 
                  'squat']
class1_dict = {exe:i for i,exe in enumerate(exercise_names)} 
class2_dict = {'correct': 1,  'incorrect': 0}
# all class = class1.val * class2.val

max_frame = 800
max_frame_seg = 160

#############################
def ske_vis(data, **kwargs):
    from dataset.skeleton import vis
    from dataset.execheck_skeleton import edge21 as edge
    vis(data, edge=edge, **kwargs)

def normalize_skeletons(skeleton, origin=0, base_bone=[0,16]):
    '''
    :param skeleton: T, V, C=(x, y, z)
    :param origin: int (default: pelvis)
    :param base_bone: [int, int] (default: pelvis-neck)
    :return: np.array
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


def read_csv_file(file_path): # segmentation file
    import csv
    data_list = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if row[0].startswith('file://'):
                path = row[0][7:] # excluding 'file://'
                frame_numbers = list(map(int, row[1:]))
                assert len(frame_numbers)==6
                data_list.append((path, frame_numbers))

    data_list.sort(key=lambda tup:tup[0]) 
    return data_list




#####################################################
def gendata(data_root, out_path, split='val'):
    val_sub = [6]
    labels = [] # (exercise_name, correctness):(int,int)
    names = [] # subID-exercise_name-correctness: str
    if split=='val' : # validation set
        sample_set = val_sub
    elif split=='train': # train set
        sample_set = list(set(range(7))-set(val_sub))
    else:
        sample_set = list(range(7))

    json_file = glob.glob(f'{data_root}/p{sample_set}/*correct_sdk.json')
    data_skeleton = np.zeros((len(json_file), 3, max_frame, num_joints, 1), dtype=np.float32)

    json_file.sort()
    for i, file in enumerate(json_file):
        body_xyz = []
        # body_ori = []
        data = json.load(open(file, 'r'))
        subject_id = os.path.basename(os.path.dirname(file))[1]            
        tmp = os.path.basename(file).split('_')
        exe_name = '_'.join(tmp[:-2])
        cls1 = class1_dict[exe_name]
        cls2 = class2_dict[tmp[-2]]
        labels.append((int(cls1), int(cls2)))
        names.append(f'{subject_id}-{exe_name}-{tmp[-2]}')

        num_frames = len(data['frames'])
        last_fid = data['frames'][-1]['frame_id']
        if num_frames != last_fid+1: # -- not always true
            print(f"{file}, {num_frames} vs {last_fid}") 

        for frame in data['frames']:
            if not frame['bodies']:
                num_frames -= 1
                print("skipped frame", frame['frame_id']) 
            else:
                body_xyz.append(frame['bodies'][0]['joint_positions'])
                # body_ori.append(frame['bodies'][0]['joint_orientations'])
        body_xyz = np.array(body_xyz)
        body_xyz = body_xyz[:, INCL, :]
        # plot_hm3d(body_xyz[10])
        # body_ori = np.array(body_ori)
        # body_ori = body_ori[:, INCL, :]
        assert body_xyz.shape == (num_frames, num_joints, 3)
        # assert body_ori.shape == (num_frames, num_joints, 4)
        # print(f"p{subject}, {exe_name}, {num_frames} frames")

        # normalize skeleton 
        normed_skeletons = normalize_skeletons(body_xyz, origin=0, base_bone=[0,16]).transpose((2,0,1)) # T,V,C -> C,T,V
        
        # # for visual check, 
        # skeletons = np.copy(normed_skeletons)
        # # # z-depth -> z-height
        # # skeletons[0,:,:], skeletons[1,:,:], skeletons[2,:,:] = skeletons[2,:,:], skeletons[0,:,:], skeletons[1,:,:]
        # skeletons = np.expand_dims(skeletons,-1) # C,T,V -> C,T,V,M
        # ske_vis(skeletons, view=1, pause=0.1)

        data_skeleton[i,:,:num_frames,:,0] = normed_skeletons 

    # ske_vis(data_skeleton[-1], view=1, pause=0.05)

    with open('{}/{}_label.pkl'.format(out_path, split), 'wb') as f:
        pickle.dump((names, labels), f)

    np.save('{}/{}_data_joint.npy'.format(out_path, split), data_skeleton) # NCTVM



def gendata_seg(seg_file, out_path): # we will split train-val later
    
    data_list = read_csv_file(seg_file)
    assert len(data_list) == 7*10*2
    skipping_log = []

    files = []
    labels = []  
    data_skeleton = np.zeros((5*len(data_list), 3, max_frame_seg, num_joints, 1), dtype=np.float32) # NCTVM (700,3,160,21,1)
    for i, (json_file, seg_list) in enumerate(data_list):
        
        if json_file.endswith('.mkv'):
            json_file = json_file.strip('.mkv')+'_sdk.json' 
        print(json_file)
        
        sub_id = os.path.basename(os.path.dirname(json_file))[1]            
        tmp = os.path.basename(json_file).split('_')
        exe_name = '_'.join(tmp[:-2])
        cls1 = class1_dict[exe_name]
        cls2 = class2_dict[tmp[-2]]
        print(f'sbuject:{sub_id}, exercise:{exe_name}, correctness:{cls2}')

        data = json.load(open(json_file, 'r'))
        body_xyz = []
        
        for frame in data['frames']:
            if not frame['bodies']:
                print("skipped frame", frame['frame_id']) 
                skipping_log.append(f"{json_file}, skip at frame = {frame['frame_id']}")
            else:
                body_xyz.append(frame['bodies'][0]['joint_positions'])
                # body_ori.append(frame['bodies'][0]['joint_orientations'])
        body_xyz = np.array(body_xyz)
        body_xyz = body_xyz[:, INCL, :] # (n_frames, 21,3) 
        # normalized skeleton 
        skeletons = normalize_skeletons(body_xyz, origin=0, base_bone=[0,16]) # TVC
        # print('skeletons shape: ', skeletons.shape)
        
        # SEGMENTING
        for rep_id, (start, end) in enumerate(zip(seg_list[:-1], seg_list[1:])):
            name = f'{sub_id}-{exe_name}-{tmp[-2]}-R{rep_id+1}'
            label = (int(cls1), int(cls2))
            # print(f'{name}:{start}-{end}, {label}')
            files.append(name)
            labels.append(label)
            # slice skeletons then transpose
            skel_seg = skeletons[start:end].transpose(2,0,1) # TVC- -> CTV 
            data_skeleton[5*i+rep_id, :, :(end-start), :, 0] = skel_seg
    
    # # visual check
    # for i in range(0,len(files),5):
    #     ske_vis(data_skeleton[i], view=1, pause=0.01, title=files[i])
    with open('{}/seg_label.pkl'.format(out_path), 'wb') as f:
        pickle.dump((files, labels), f)

    np.save('{}/seg_data_joint.npy'.format(out_path), data_skeleton)

    with open(os.path.join(out_path, 'skipped_frames.txt'), "w") as f:
        for row in skipping_log:
            f.write(f'{row}\n')
    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ExerciseCheck Data Converter.')
    parser.add_argument('--data_root', default='/research/yiwen/DATA/ExeCheck')
    parser.add_argument('--seg_file', default='/research/yiwen/DATA/ExeCheck/RepSeg.csv')
    parser.add_argument('--out_folder', default='../../processed_execheck/')

    args = parser.parse_args()

    out_path = args.out_folder
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    # for split in ['val', 'train']:
    #     print(f'gendata for {split}')
    #     gendata(args.data_root,out_path,split)
    
    gendata_seg(args.seg_file, out_path)

    