import numpy as np
import pickle as pkl
import random
import os

root = '../../processed_execheck'
label_path = f'{root}/seg_label.pkl'
sample_name, labels = pkl.load(open(label_path, 'rb'))
data_path = f'{root}/seg_data_joint.npy'
data = np.load(data_path)
num_total = len(data)


def get_val_ids_xsub(val_sub):
    val_sub = str(val_sub)
    val_ids = []
    for i, sample in enumerate(sample_name):
        if sample.startswith(val_sub):
            val_ids.append(i)

    return val_ids

def get_val_ids_random(mode):

    if mode == 'psub': # split by subject (1/7): train:val = 6:1 = ~86:~14
        val_sub = str(6) # subject 6 as validation
        val_ids = []
        for i, sample in enumerate(sample_name):
            if sample.startswith(val_sub):
                val_ids.append(i)
        # keep 2/5 reps from val, s.t more training (2/5/7), train:val = 94:6
        partial_val_ids = []
        rep_list = list(range(5))
         # keep correct vs incorrect paired
        for i in range(10):
            v1, v2 = random.sample(rep_list, k=2)
            # print(v1,v2)
            partial_val_ids.append(val_ids[i*10+v1])
            partial_val_ids.append(val_ids[i*10+v2])
            partial_val_ids.append(val_ids[i*10+v1+5])
            partial_val_ids.append(val_ids[i*10+v2+5])
        # print(partial_val_ids)
        return partial_val_ids

    elif mode == 'random': # split in random (2*20/700), train:val = 94:6
        random.seed(9) 
        num_vals_per_exe = 2  # two segs per exercise in val
        sub_list = list(range(7))
        rep_list = list(range(1,6))
        exe_list = ['arm_circle', 
                    'forward_lunge', 
                    'high_knee_raise', 
                    'hip_abduction', 
                    'leg_extension', 
                    'shoulder_abduction', 
                    'shoulder_external_rotation', 
                    'shoulder_flexion', 
                    'side_step_squat', 
                    'squat']
        val_name = []
        sub = random.choices(sub_list, k=num_vals_per_exe*len(exe_list))    
        rep = random.choices(rep_list, k=num_vals_per_exe*len(exe_list))    
        assert len(sub) == len(rep)
        # print('sub: ', sub)
        # print('rep: ', rep)
        for sub_id, rep_id, exe_name in zip(sub, rep, exe_list*num_vals_per_exe):
            val_name.append(f'{sub_id}-{exe_name}-correct-R{rep_id}')
            val_name.append(f'{sub_id}-{exe_name}-incorrect-R{rep_id}')

        val_ids = [sample_name.index(name) for name in val_name]
        # print(val_name)
        # print(val_ids)
        return val_ids
    
# val_ids = get_val_ids('psub')
# train_ids = list(set(range(num_total))-set(val_ids))
# train_data = data[train_ids]
# print(train_data.shape)
# exit()

# for mode in ['xsub', 'psub', 'random']:
for val_sub in range(6):
    out_path = f'{root}/xsub_v{val_sub}/'
    os.makedirs(out_path, exist_ok=True)
    val_ids = get_val_ids_xsub(val_sub)
    val_files, val_labs = [], []
    for val_id in val_ids:
        val_files.append(sample_name[val_id]) 
        val_labs.append(labels[val_id])
    val_data = data[val_ids]
    
    train_ids = list(set(range(num_total))-set(val_ids))
    train_files, train_labs = [], []
    for train_id in train_ids:
        train_files.append(sample_name[train_id]) 
        train_labs.append(labels[train_id])
    train_data = data[train_ids]

    assert len(train_labs)+len(val_labs) == num_total
    
    pkl.dump((val_files, val_labs), open('{}/seg_label_val.pkl'.format(out_path), 'wb'))
    np.save('{}/seg_data_joint_val.npy'.format(out_path), val_data)
    pkl.dump((train_files, train_labs), open('{}/seg_label_train.pkl'.format(out_path), 'wb'))
    np.save('{}/seg_data_joint_train.npy'.format(out_path), train_data)