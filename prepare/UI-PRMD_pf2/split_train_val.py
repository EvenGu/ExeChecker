import numpy as np
import pickle as pkl
import random
import os

root = 'processed_uiprmd' # python path is poject folder
label_path = f'{root}/seg_label.pkl'
sample_name, labels = pkl.load(open(label_path, 'rb'))
data_path = f'{root}/seg_data_joint.npy'
data = np.load(data_path)
num_total = len(data) # 1998


# sample_name: sub[-3:]}-{exec}-{correctness}-R{rep_id:02d}
exe_set = set([name.split('-')[1] for name in sample_name])


def get_val_ids_xsub(val_sub):
    # val_sub = '010' # subject010 as validation
    val_sub = f'{val_sub:03d}'
    val_ids = []
    for i, sample in enumerate(sample_name):
        if sample.startswith(val_sub):
            val_ids.append(i)
    return val_ids
  
def get_val_ids_random(mode):

    if mode == 'random': # split in random, 
        random.seed(9) 
        num_vals_per_exe = 5  # 5 segs per exercise in val
        sub_list = list(range(1,11))
        rep_list = list(range(1,10))
        exe_list = list(exe_set)
        val_name = []
        sub = random.choices(sub_list, k=num_vals_per_exe*len(exe_list))    
        rep = random.choices(rep_list, k=num_vals_per_exe*len(exe_list))    

        for sub_id, rep_id, exe_name in zip(sub, rep, exe_list*num_vals_per_exe):
            val_name.append(f'{sub_id:03d}-{exe_name}-correct-R{rep_id:02d}')
            val_name.append(f'{sub_id:03d}-{exe_name}-incorrect-R{rep_id:02d}')

        val_ids = [sample_name.index(name) for name in val_name]
        # print(val_name)
        # print(val_ids)
        return val_ids
    

for val_sub in range(10):
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