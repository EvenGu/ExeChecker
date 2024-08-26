import torch
from torch.nn import CrossEntropyLoss, TripletMarginLoss, TripletMarginWithDistanceLoss
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, CyclicLR

import numpy as np
import setproctitle
import os
from tqdm import tqdm
import time

from utility.average_meter import AverageMeter
from utility.log import TimerBlock, IteratorTimer
from tensorboardX import SummaryWriter

from train_val_test import parser_args
from utility.utils import to_onehot, mixup

from method_choose.data_choose import data_choose, init_seed
# from method_choose.loss_choose import loss_choose
from method_choose.lr_scheduler_choose import lr_scheduler_choose
from method_choose.model_choose import model_choose, rm_module
from method_choose.optimizer_choose import optimizer_choose

from loss.loss import TripletLoss_ratio
from loss.hierarchical_loss import HierarchicalLoss
from dataset.execheck_skeleton import hierarchy


def val_multi(triplet_loader, model, criterions):
    process = tqdm(IteratorTimer(triplet_loader), desc='val Multi: ', dynamic_ncols=True)
    
    trip_feature_losses = AverageMeter('feat_triplet_loss')
    trip_accuracy = AverageMeter('trip_acc')
    
    model.eval()

    for bi, (anchors, positives, negatives) in enumerate(process):
        with torch.no_grad():
            _, feat_a = model(anchors[0].cuda())
            _, feat_p = model(positives[0].cuda())
            _, feat_n = model(negatives[0].cuda())
            n = feat_a.size()[0]

            ls_triplet_feat = criterions["trip_feat_fn"](feat_a, feat_p, feat_n) 

            dist_a2p = torch.norm(feat_a - feat_p, p=2, dim=1)
            dist_a2n = torch.norm(feat_a - feat_n, p=2, dim=1)
            pred = (dist_a2p / (dist_a2n + 1e-8)).cpu().data
            trip_acc = (pred<1).sum()*1.0/n
            
        trip_feature_losses.update(ls_triplet_feat.data.item())
        trip_accuracy.update(trip_acc)   

    process.close() 
    return trip_feature_losses.avg, trip_accuracy.avg
            
#####################################################
def count_params(model): # does not work with LazyLinear in v3
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optim, epoch, global_step, args, suffix=None):    
    checkpoint = {
        'model': model.state_dict(),
        'optimizers': optim.state_dict(),
        'epoch': epoch,
        'global_step': global_step
    }
    
    filepath = f'{args.model_saved_name}_e-{epoch}_{global_step}.pth'
    if suffix:
        filepath = f'{args.model_saved_name}_{suffix}_e-{epoch}_{global_step}.pth'

    torch.save(checkpoint, filepath)
#####################################################

with TimerBlock("Triplet Training") as block:

    args = parser_args.parser_args(block)
    assert args.task == 'multi'

    init_seed(7)
    setproctitle.setproctitle(args.model_saved_name)
    block.log(f'CUDA_VISIBLE_DEVICES: {os.getenv("CUDA_VISIBLE_DEVICES")}')
    block.log('work dir: ' + args.model_saved_name)
    
    train_writer = SummaryWriter(os.path.join(args.model_saved_name, 'train'), 'train')
    val_writer = SummaryWriter(os.path.join(args.model_saved_name, 'val'), 'val')

    triplet_loader_train, triplet_loader_val = data_choose(args, block)
    triplet_loader = triplet_loader_train
    n_iters_train = len(triplet_loader)
 
    # cls_fn = CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # triplet_fn = TripletLoss_ratio(swap=True)
    triplet_fn = TripletLoss_ratio(swap=True, dist_func=lambda x, y: 1.0 - F.cosine_similarity(x, y))
    criterions = {'trip_feat_fn': triplet_fn,
                  'cls_fn': None}
    
    global_step, start_epoch, multi_model, optimizer_dict = model_choose(args, block)

    optimizer = optimizer_choose(multi_model, args, block)
    # if optimizer_dict is not None and args.last_model is not None:
    #     try:
    #         optimizer.load_state_dict(optimizer_dict)
    #         block.log('load optimizer from state dict')
    #     except:
    #         block.log('optimizer not matched')
    # else:
    #     block.log('no pretrained optimizer is loaded')

    # lr_scheduler = CosineAnnealingLR(optimizer, T_max=n_iters_train*2, eta_min=1e-5)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    lr_scheduler = ExponentialLR(optimizer, last_epoch=-1, gamma=args.lr_decay_ratio) # reset sched

    block.log('Model total number of params: ' +str(count_params(multi_model)))

    best_step = 0
    best_epoch = 0
    best_loss = np.inf
    best_step_val = 0
    best_epoch_val = 0
    best_loss_val = np.inf
    best_loss_val_acc = 0
    
    if args.val_first:

        trip_ls_feat_val, trip_acc_val = val_multi(triplet_loader_val, multi_model, criterions)
        
        block.log(f'Init at e-{start_epoch}, step-{global_step}: trip_Acc: {trip_acc_val: .4f}, \
                  trip_Loss: {trip_ls_feat_val}.')
        best_loss_val = trip_ls_feat_val
        best_loss_val_acc = trip_acc_val

    block.log(f'num_steps in training per epoch: {n_iters_train}')
    # start_epoch = 0
    block.log('start epoch {} -> max epoch {}\n'.format(start_epoch, args.max_epoch))

    for epoch in tqdm(range(start_epoch, args.max_epoch)):

        # lr = optimizer.param_groups[0]['lr']
        # block.log('Epoch : {}, Current lr: {}'.format(epoch, lr))

        for key, value in multi_model.named_parameters():
            value.requires_grad = True
        for freeze_key, freeze_epoch in args.freeze_keys:
            if freeze_epoch > epoch:
                block.log('{} is froze'.format(freeze_key))
                for key, value in multi_model.named_parameters():
                    if freeze_key in key:
                        # block.log('{} is froze'.format(key))
                        value.requires_grad = False

        last_epoch_time = time.time()
        multi_model.train()  # set model to training mode !!
        process = tqdm(IteratorTimer(triplet_loader), desc='triplets loader, train Multi: ', dynamic_ncols=True)        
        trip_feat_losses = AverageMeter('feat_triplet_loss')
        trip_accuracy = AverageMeter('trip_acc')

        for batch_idx, (anchors, positives, negatives) in enumerate(process):

            # if batch_idx == 0 and epoch == 0:
            #     train_writer.add_graph(multi_model, anchors[0].cuda())

            optimizer.zero_grad()

            _, feat_a = multi_model(anchors[0].cuda())
            _, feat_p = multi_model(positives[0].cuda())
            _, feat_n = multi_model(negatives[0].cuda())


            n = feat_a.size()[0]

            ls_triplet_feat = criterions["trip_feat_fn"](feat_a, feat_p, feat_n) 

            dist_a2p = torch.norm(feat_a - feat_p, p=2, dim=1)
            dist_a2n = torch.norm(feat_a - feat_n, p=2, dim=1)
            pred = (dist_a2p / (dist_a2n + 1e-8)).cpu().data
            trip_acc = (pred<1).sum()*1.0/n

            ls_triplet_feat.backward() #retain_graph=True

            # # for debug
            # for name, param in multi_model.named_parameters():
            #     if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
            #         print(f"NaN or Inf found in gradients of {name}")

            optimizer.step()
            
            if (batch_idx+1) % (n_iters_train//2) == 0:
                lr_scheduler.step()

            global_step += 1
            
            ls1 = ls_triplet_feat.data.item()
            trip_feat_losses.update(ls1) 
            trip_accuracy.update(trip_acc)   

            process.set_description(
                f"Epoch {epoch}, Train loss, acc: {ls1:.4f}, {trip_acc:.3f}. ")

            # log per batch
            train_writer.add_scalars('triplet_training', {'feats_loss': ls1,
                                                'accuracy': trip_acc}, global_step)
            # lr = optimizer.param_groups[0]['lr']
            train_writer.add_scalar('lr', lr_scheduler.get_last_lr()[-1], global_step)
            
            if global_step % args.num_step_per_validate == 0:

                trip_ls_feat_val, trip_acc_val = val_multi(triplet_loader_val, multi_model, criterions)

                block.log(f"    Step{global_step}: Mean train/val trip_feat_loss: {trip_feat_losses.avg:.4f}/{trip_ls_feat_val:.4f}, "
                          f" trip_feat accuracy: {trip_acc:.3f}/{trip_acc_val:.3f}"
                          )

                val_writer.add_scalars('triplet_training', {'feats_loss': trip_ls_feat_val,
                                                'accuracy': trip_acc_val}, global_step)


                if trip_ls_feat_val < best_loss_val:
                    best_loss_val = trip_ls_feat_val
                    best_epoch_val = epoch
                    best_step_val = global_step
                    save_checkpoint(multi_model, optimizer, epoch, global_step, args, suffix='bestVal')
                    block.log(f'save checkpoint for bestVal {epoch}-{global_step}, loss: {trip_ls_feat_val:.4f}, cls_acc: {trip_acc_val:.4f})')
                
                multi_model.train()  # set model back to training mode !!


        process.close() 
        # train_writer.add_scalar('epoch_time', time.time() - last_epoch_time, epoch)
        
        trip_ls_feat_val, trip_acc_val = val_multi(triplet_loader_val, multi_model, criterions)

        block.log(f"Epoch {epoch}: Mean train/val trip_feat_loss: {trip_feat_losses.avg:.4f}/{trip_ls_feat_val:.4f}, "
                    f" trip_feat accuracy: {trip_acc:.3f}/{trip_acc_val:.3f}"
                    )

        val_writer.add_scalars('triplet_training', {'feats_loss': trip_ls_feat_val,
                                                'accuracy': trip_acc_val}, global_step)


        # save every n_epochs
        if (epoch + 1) % args.num_epoch_per_save == 0:
            save_checkpoint(multi_model, optimizer, epoch, global_step, args)

        # save if better
        if trip_ls_feat_val < best_loss_val:
            best_loss_val = trip_ls_feat_val
            best_epoch_val = epoch
            best_step_val = global_step
            save_checkpoint(multi_model, optimizer, epoch, global_step, args, suffix='bestVal')

        block.log(f'Training finished for epoch {epoch} (time:{time.time()-last_epoch_time:.3f}s)\n')

    block.log(f'Best model VAL: {args.model_saved_name}_e-{best_epoch_val}_{best_step_val}.pth, loss: {best_loss_val}')
    block.save(os.path.join(args.model_saved_name, 'log.txt'))    
