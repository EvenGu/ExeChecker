from __future__ import print_function, division
import torch
import torch.nn as nn
from collections import OrderedDict
import shutil
import inspect
import os

def rm_module(old_dict):
    new_state_dict = OrderedDict()
    for k, v in old_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def model_choose(args, block):
    m = args.model # the backbone
    if m == 'st2ransformer_dsta':
        from model.st2ransformer_dsta import DSTANet
        module = DSTANet(num_class=args.class_num, num_point=args.joints_num, skeleton=args.data, **args.model_param)
        # shutil.copy2(inspect.getfile(DSTANet), args.model_saved_name)
    elif m == 'stgat':
        from model.stgat.st_gat import DSTANet_smp
        module = DSTANet_smp(num_point=args.joints_num, skeleton=args.data, **args.model_param)
    elif m == 'stgcn':
        from model.stgcn.st_gcn import STGCN
        graph_args={"layout": args.data, "strategy": 'spatial'}
        module = STGCN(graph_args, edge_importance_weighting=True, **args.model_param)
    elif m == 'decoupled':
        pass
    else:
        raise (NotImplementedError(f"No module {m}"))
    
    block.log(f'Model (Backbone) load finished: {args.model}')

    # TODO task:
    if args.task == 'multi': 
        if args.triplet_model == 'v5':
            from model.taskModels import MultiTaskModel_v5 as MultiModel
            model = MultiModel(graph_net=module, num_class=args.class_num, **args.triplet_model_param)
        elif args.triplet_model == 'v4':
            from model.taskModels import MultiTaskModel_v4 as MultiModel
            model = MultiModel(graph_net=module, num_class=args.class_num, **args.triplet_model_param)
        elif args.triplet_model == 'v3':
            from model.taskModels import MultiTaskModel_v3 as MultiModel
            model = MultiModel(graph_net=module, num_class=args.class_num, **args.triplet_model_param)
        elif args.triplet_model == 'v2':
            from model.taskModels import MultiTaskModel_v2 as MultiModel
            model = MultiModel(graph_net=module, num_joints=args.joints_num, **args.triplet_model_param)
        else: # the naive one
            from model.taskModels import MultiTaskModel as MultiModel
            model = MultiModel(graph_net=module, **args.triplet_model_param)
        block.log(f'load TripletModel_{args.triplet_model} for {args.task} task')
    elif args.task == 'h_classify':
        from model.taskModels import Hierachical_Classification_Model as HCM
        model = HCM(graph_net=module, num_class=args.class_num)
        block.log(f'load Hierachical_Classifation_Model for {args.task} task.')
    else: # args.task == 'classify': 
        from model.taskModels import Classifier
        model = Classifier(net=module, num_class=args.class_num)
        block.log(f'load {args.model} for classification task.')

    if args.pre_trained_model is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.pre_trained_model)  # ['state_dict']
        if type(pretrained_dict) is dict:
            # if ('optimizer' in pretrained_dict.keys()):
            #     optimizer_dict = pretrained_dict['optimizer']
            pretrained_dict = pretrained_dict['model']
        pretrained_dict = rm_module(pretrained_dict)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        keys = list(pretrained_dict.keys())
        for key in keys:
            for weight in args.ignore_weights:
                if weight in key:
                    if pretrained_dict.pop(key) is not None:
                        block.log('Sucessfully Remove Weights: {}.'.format(key))
                    else:
                        block.log('Can Not Remove Weights: {}.'.format(key))
        block.log('following weight not load: ' + str(set(model_dict) - set(pretrained_dict)))
        model_dict.update(pretrained_dict)
        # block.log(model_dict)
        model.load_state_dict(model_dict)
        block.log('Pretrained model load finished: ' + args.pre_trained_model)

    # mutually excl w/ pretrained
    global_epoch = 0
    global_step = 0
    optimizers = None
    if args.last_model is not None:
        model_dict = model.state_dict()
        ckpt = torch.load(args.last_model)        
        pretrained_dict = rm_module(ckpt['model'])
        block.log('Following weights will not load from last model: ' + str(set(model_dict) - set(pretrained_dict)))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        global_epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('global_step', 0)
        optimizers = ckpt.get('optimizers')
        block.log(f'load checkpoints from {args.last_model}')

        # naming rule: 
        # name for the original classification model: **-{epoch}-{glob_step}.state
        # name for the triplet multi_model: **_e-{epoch}_(loc_step).pth
        if global_epoch == global_step == 0:
            name = os.path.splitext(os.path.basename(args.last_model))[0]
            try:
                global_step = int(name.split('_')[-1])
            except:
                global_step = 0
            try:    
                global_epoch = int(name.split('_')[-2][2:]) # e-X
            except:
                global_epoch = 0
        block.log(f'continue training from epoch {global_epoch}, global step {global_step}')

    model.cuda()
    model = nn.DataParallel(model, device_ids=args.device_id)
    block.log(f'copy model to gpu, device_id: {args.device_id}')

    # shutil.copy2(__file__, args.model_saved_name)

    return global_step, global_epoch, model, optimizers

