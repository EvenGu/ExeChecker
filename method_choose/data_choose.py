from __future__ import print_function, division

from torch.utils.data import DataLoader

import torch
import numpy as np
import random
import shutil
import inspect

def init_seed(x):
    torch.cuda.manual_seed_all(x) # for GPU
    torch.manual_seed(x) # for CPU
    np.random.seed(x)
    random.seed(x)
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# TODO about 'block', restructure
def data_choose(args, block):
    
    if args.data == 'execheck_skeleton':
        from dataset.execheck_skeleton import EXECHECK_SKE as SKE
    elif args.data == 'uiprmd_skeleton':
        from dataset.uiprmd_skeleton import UIPRMD_SKE as SKE
    else:
        raise (NotImplementedError(f'No data loader for {args.data}'))
    
    if isinstance(args.class_num, list):
        data_set_train = SKE(split='train', use_hierachy=True, folder_path=args.folder_path, num_joints=args.joints_num, **args.data_param['train_data_param'])
        data_set_val = SKE(split='val', use_hierachy=True, folder_path=args.folder_path, num_joints=args.joints_num, **args.data_param['val_data_param'])
    else:
        if args.class_num == 20:
            data_set_train = SKE(split='train', use_subclass=True, folder_path=args.folder_path, num_joints=args.joints_num, **args.data_param['train_data_param'])
            data_set_val = SKE(split='val', use_subclass=True, folder_path=args.folder_path, num_joints=args.joints_num, **args.data_param['val_data_param'])
        else: # 10 classes
            data_set_train = SKE(split='train', folder_path=args.folder_path, num_joints=args.joints_num, **args.data_param['train_data_param'])
            data_set_val = SKE(split='val', folder_path=args.folder_path, num_joints=args.joints_num, **args.data_param['val_data_param'])

    
    if args.task == 'classify' or args.task == 'h_classify':
        data_loader_train = DataLoader(data_set_train, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.worker, drop_last=False, pin_memory=args.pin_memory,
                                        worker_init_fn=init_seed)
        data_loader_val = DataLoader(data_set_val, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.worker, drop_last=False, pin_memory=args.pin_memory,
                                    worker_init_fn=init_seed)
        
    elif args.task == 'multi':
        from dataset.triplets_loader import TripletDataset
        triplet_dataset_train = TripletDataset(data_set_train, split="train", exec_class=args.exercise_class,   
                                    folder_path=args.folder_path)
        triplet_dataset_val = TripletDataset(data_set_val, split="val", exec_class=args.exercise_class,   
                                    folder_path=args.folder_path)
        
        data_loader_train = DataLoader(triplet_dataset_train, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.worker, pin_memory=args.pin_memory)
        data_loader_val = DataLoader(triplet_dataset_val, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.worker, pin_memory=True)
    

    block.log(f'Data load finished: {args.data} for {args.task} task on exercises {args.exercise_class}')

    # shutil.copy2(__file__, args.model_saved_name)
    return data_loader_train, data_loader_val
