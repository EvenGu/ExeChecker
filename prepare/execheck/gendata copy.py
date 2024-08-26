import argparse
import pickle, json
import numpy as np
import os, glob
from tqdm import tqdm
import sys
sys.path.extend(['../../', '../'])

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


####################### Meta Data ############################################
INCL = [0, 1, 2, 3, 5, 6, 7, 12, 13, 14, 18, 19, 20, 22, 23, 24, 27, 
        8, 15, 21, 25] # left right hand foot
num_joints = len(INCL)

JOINTS = ['PELVIS', 'SPINE_NAVEL', 'SPINE_CHEST', 'NECK',           # 0-3
		'SHOULDER_LEFT', 'ELBOW_LEFT', 'WRIST_LEFT',                # 4-6
		'SHOULDER_RIGHT', 'ELBOW_RIGHT', 'WRIST_RIGHT',             # 7-9
		'HIP_LEFT', 'KNEE_LEFT', 'ANKLE_LEFT',                      # 10-12
		'HIP_RIGHT', 'KNEE_RIGHT', 'ANKLE_RIGHT',                   # 13-15
		'NOSE',                                                     # 16
		'HAND_LEFT', 'HAND_RIGHT', 'FOOT_LEFT', 'FOOT_RIGHT']

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

############################# butterworth filter ##############

from scipy.signal import butter, filtfilt, sosfiltfilt
order = 4
cutoff = 7.5
# b, a = butter(order, cutoff, fs=30, btype='low')
sos = butter(order, cutoff, fs=30, btype='low', output='sos')
sos15 = butter(order, 14.9, fs=30, btype='low', output='sos')
# filtered_data = filtfilt(b, a, data)
################################################
def plot_jointXYZ(joint_pos, title=None, save_path=None):
    """
    joint_pos: np.ndarray of shape (N, 3)
    """
    num_frame = joint_pos.shape[0]
    time = np.arange(num_frame)
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot x, y, z coordinates
    axs[0].plot(time, joint_pos[:, 0], label='x')
    axs[1].plot(time, joint_pos[:, 1], label='y')
    axs[2].plot(time, joint_pos[:, 2], label='z')

    # Set labels and titles
    axs[0].set_ylabel('X Position')
    axs[1].set_ylabel('Y Position')
    axs[2].set_ylabel('Z Position')
    axs[2].set_xlabel('Frame Number')

    axs[0].set_title(f'{title} jointXYZ Over Time')

    # Add grid for better visualization
    for ax in axs:
        ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


#############################

def calculate_rotation_matrix(joints, y_bone=[0,3], x_bone=[13,10]):
    """
    Calculates the rotation matrix to align the body pose to the frame
    ( upright and frontal )

    Parameters:
    - joints: np.array of shape (V,3) where V=21 in exeCheck
    - y_bone: [int, int]  (default: pelvis -> neck)
    - x_bone: [int, int]  (default: hip_r -> hip_l)

    Returns:
    - rotation_matrix: np.ndarray of shape (3, 3), the rotation matrix.
    """

    assert joints.shape[-1] == 3
    # Define the target vectors
    y_axis = np.array([0, 1, 0])  # Target direction for the base bone
    x_axis = np.array([1, 0, 0])  # Target direction for the second bone

    # Calculate the current direction of the base bone (joint[3] - joint[0])
    base_bone = joints[y_bone[1]] - joints[y_bone[0]]
    base_bone /= np.linalg.norm(base_bone)  # Normalize the vector

    # Calculate the rotation to align base_bone with the y-axis
    rot_to_y = R.align_vectors([y_axis], [base_bone])[0]
    
    # Rotate all joints using rot_to_y
    rotated_joints = rot_to_y.apply(joints)

    # Calculate the current direction of the x_bone (joint[10] - joint[13]) after the first rotation
    second_bone = rotated_joints[x_bone[1]] - rotated_joints[x_bone[0]]
    second_bone /= np.linalg.norm(second_bone)  # Normalize the vector

    # Calculate the rotation needed to align second_bone with the x-axis
    rot_to_x = R.align_vectors([x_axis], [second_bone])[0]

    # Combine the two rotations
    rotation_matrix = rot_to_x.as_matrix() @ rot_to_y.as_matrix()

    return rotation_matrix

def normalize_skeletons_and_align_to_frame(skeleton, origin=0, base_bone=[0,3], rot_mat=None):
    '''
    :param skeleton: T, V, C=(x, y, z)
    :param origin: int (default: pelvis)
    :param base_bone: [int, int] (default: pelvis -> neck)
    :param rot_mat: np.ndarray of shape (3, 3) --if provided, apply the rot_mat for all poses
    :return: np.array
    '''

    T, V, C = skeleton.shape

    if rot_mat is not None:
        skeleton = np.einsum('ij,tvj->tvi', rot_mat, skeleton)

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

def ske_vis(data, **kwargs):
    from dataset.skeleton import vis
    from dataset.execheck_skeleton import edge21 as edge
    vis(data, edge=edge, **kwargs)


#####################################################

def gendata_seg(seg_file, out_path): # we will split train-val later
    """
    1. re-orient xyz on the first frame, 
    2. segment, 
    """
    data_list = read_csv_file(seg_file)
    assert len(data_list) == 7*10*2
    skipping_log = []

    labels = [] # (exercise_name, correctness):(int,int)
    names = [] # subID-exercise_name-correctness-repID: str
    data_skeleton = np.zeros((5*len(data_list), 3, max_frame_seg, num_joints, 1), dtype=np.float32) # NCTVM (700,3,160,21,1)
    for i, (json_file, seg_list) in enumerate(data_list):
        
        if json_file.endswith('.mkv'):
            json_file = json_file.strip('.mkv')+'_sdk.json' 
        print(json_file)
        
        sub_id = os.path.basename(os.path.dirname(json_file))[1]   
        
        # # small trials
        # if sub_id != '6':
        #     continue

        tmp = os.path.basename(json_file).split('_')
        exe_name = '_'.join(tmp[:-2])
        cls1 = class1_dict[exe_name]
        cls2 = class2_dict[tmp[-2]]
        print(f'subject:{sub_id}, exercise:{exe_name}, correctness:{cls2}')

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
        
        # normalize skeleton and align
        xyz0 = body_xyz[0,:,:].copy()
        rot_mat = calculate_rotation_matrix(xyz0, y_bone=[0,3], x_bone=[13,10])
        skeletons = normalize_skeletons_and_align_to_frame(body_xyz, rot_mat=rot_mat) # TVC
        
        T,V,C = skeletons.shape

        # low-pass filter butterworth
        filtered_ske_sos = np.zeros_like(skeletons)
        filtered_ske_ba = filtered_ske_sos.copy()
        for v in range(V):
            for c in range(C):
                filtered_ske_sos[:,v,c] = sosfiltfilt(sos, skeletons[:,v,c])
                filtered_ske_ba[:,v,c] = sosfiltfilt(sos15, skeletons[:,v,c])
        # plotting for QA check
        save_folder = f'{out_path}/{exe_name}_p{sub_id}_{tmp[-2]}'
        os.makedirs(save_folder, exist_ok=True)
        for i, j_name in enumerate(JOINTS):
            # plot_jointXYZ(skeletons[:,i,:], title=j_name, save_path=f'{save_folder}/{j_name}.png')
            plot_jointXYZ(filtered_ske_sos[:,i,:], title=j_name, save_path=f'{save_folder}/{j_name}_sos.png')
            # plot_jointXYZ(filtered_ske_ba[:,i,:], title=j_name, save_path=f'{save_folder}/{j_name}_sos15.png')


        # SEGMENTING
        for rep_id, (start, end) in enumerate(zip(seg_list[:-1], seg_list[1:])):
            name = f'{sub_id}-{exe_name}-{tmp[-2]}-R{rep_id+1}'
            label = (int(cls1), int(cls2))
            # print(f'{name}:{start}-{end}, {label}')
            names.append(name)
            labels.append(label)
            # slice skeletons then transpose
            # skel_seg = skeletons[start:end].transpose(2,0,1) # TVC- -> CTV 
            skel_seg = filtered_ske_sos[start:end].transpose(2,0,1) # TVC- -> CTV 
            data_skeleton[5*i+rep_id, :, :(end-start), :, 0] = skel_seg
    
    # # visual check
    # for i in range(0,len(names),5):
    #     ske_vis(data_skeleton[i], view=1, pause=0.01, title=names[i])


    with open('{}/seg_label.pkl'.format(out_path), 'wb') as f:
        pickle.dump((names, labels), f)

    np.save('{}/seg_data_joint.npy'.format(out_path), data_skeleton)

    # with open(os.path.join(out_path, 'skipped_frames.txt'), "w") as f:
    #     for row in skipping_log:
    #         f.write(f'{row}\n')
    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ExerciseCheck Data Converter.')
    parser.add_argument('--data_root', default='/research/yiwen/DATA/ExeCheck')
    parser.add_argument('--seg_file', default='/research/yiwen/DATA/ExeCheck/RepSeg.csv')
    parser.add_argument('--out_folder', default='../../processed_execheck_re_butter')

    args = parser.parse_args()

    out_path = args.out_folder #+'_butter'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    gendata_seg(args.seg_file, out_path)

    