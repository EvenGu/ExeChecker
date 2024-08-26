import torch
from utility.log import TimerBlock
import torch.nn.functional as func
# import shutil
# import inspect
from loss.loss import *

def get_loss_function(loss_name, args):
    if loss_name == 'cross_entropy':
        return torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif loss_name == 'ctc':
        p = args.loss_param
        return CTC(p.input_len, p.target_len)
    elif loss_name == 'multi_cross_entropy':
        return multi_cross_entropy_loss()
    elif loss_name == 'mse':
        return torch.nn.MSELoss()
    elif loss_name == 'l1loss':
        return L1()
    elif loss_name == 'l2loss':
        return L2()
    elif loss_name == 'margin_rank':
        return torch.nn.TripletMarginWithDistanceLoss(swap=True)
    elif loss_name == 'triplet_naive':
        return TripletLoss(margin=args.triplet_loss_param['initial_margin'])
    elif loss_name == 'triplet_adaptive':
        return AdaptiveTripletLoss(**args.triplet_loss_param)
    elif loss_name == 'triplet_hard':
        return HardTripletLoss(margin=args.triplet_loss_param['initial_margin'])
    elif loss_name == 'hierarchial': 
        from loss.hierarchical_loss import HierarchicalLoss
        if args.data == 'execheck_skeleton':
            from dataset.execheck_skeleton import hierarchy
        elif args.data == 'uiprmd_skeleton':
            from dataset.execheck_skeleton import hierarchy
        else:
            raise NotImplementedError
        return HierarchicalLoss(hierarchy)
    else:
        raise NotImplementedError


def loss_choose(args, block):

    loss_functions = {}

    # for classification
    loss = args.loss 
    cls_losses = []
    if isinstance(loss, list):
        for loss_name in loss:
            loss_fn = get_loss_function(loss_name, args)
            cls_losses.append(loss_fn)    
    else:
        cls_losses.append(get_loss_function(loss, args))
    loss_functions['cls_loss_fn'] = cls_losses
    
    block.log(f'loss functions used for classification: {loss}')

    # for triplets
    if args.task == 'multi' or args.task == 'triplet_only':
        triplet_loss = args.triplet_loss
        if isinstance(triplet_loss, list):
            triplet_losses = [get_loss_function(f'triplet_{tls}', args) for tls in triplet_loss]
        else: # naive 
            triplet_losses = [get_loss_function(f'triplet_{triplet_loss}',args)]

        loss_functions['triplet_loss_fn'] = triplet_losses

        block.log(f'loss functions used for triplets: {triplet_loss}')

    # shutil.copy2(inspect.getfile(loss_function), args.model_saved_name)
    # shutil.copy2(__file__, args.model_saved_name)
    return loss_functions


if __name__ == '__main__':
    res_ctc = torch.Tensor([[[0, 0, 1]], [[0.5, 0.6, 0.2]], [[0, 0., 1.]]])
    b = 1  # batch
    c = 2  # label有多少类
    in_len = 3  # 预测每个序列有多少个label
    label_len = 1  # 实际每个序列有多少个label

    # res_ctc = torch.rand([in_len, b, c+1])
    target = torch.zeros([b*label_len,], dtype=torch.long)

    loss_ctc = CTC(in_len, label_len)
    ls_ctc = loss_ctc(res_ctc, target)

    # loss_ctcp = CTCP(in_len, label_len)
    # ls_ctcp = loss_ctcp(res_ctc, target)

    # print(ls_ctc, ls_ctcp)
