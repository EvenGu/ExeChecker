import torch
import torch.nn as nn
import torch.nn.functional as func

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()

    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue


class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()

    def forward(self, output, target):
        lossvalue = torch.norm(output - target, p=2, dim=1).mean()
        return lossvalue


class multi_cross_entropy_loss(nn.Module):
    def __init__(self):
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        super(multi_cross_entropy_loss, self).__init__()

    def forward(self, inputs, target):
        '''
            :param inputs: N C S, 
                N: batch size, C: number of classes, S: number of predictions per sample.
            :param target: N C
            :return: 
            '''
        num = inputs.shape[-1]
        inputs_splits = torch.chunk(inputs, num, dim=-1)
        loss = self.loss(inputs_splits[0].squeeze(-1), target)
        for i in range(1, num):
            loss += self.loss(inputs_splits[i].squeeze(-1), target)
        loss /= num
        return loss

class CTC(nn.Module): # repetition segmentation?
    def __init__(self, input_len, target_len, blank=0):
        super(CTC, self).__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction='mean', zero_infinity=True)
        self.input_len = input_len
        self.target_len = target_len

    def forward(self, input, target):
        """
        :param input: T N C
            T: input sequence length, N: batch size, C: the number of classes.
        :param target: N, 
        :return:
        """
        batch_size = target.shape[0]
        input_ = torch.cat([input[:,:,-1:], input[:,:,:-1]], dim=-1).clone()
        target_ = target + 1
        target_ = target_.unsqueeze(-1)
        # target = torch.cat([target.unsqueeze(-1), target.unsqueeze(-1)], dim=1)
        ls = self.ctc(input_.log_softmax(2), target_, [self.input_len]*batch_size, [self.target_len]*batch_size)
        return ls


### triplet loss: naive, hard, adaptive
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, swap=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.swap = swap

    def forward(self, anchor, positive, negative):
        pos_dist = func.pairwise_distance(anchor, positive, p=2)
        neg_dist = func.pairwise_distance(anchor, negative, p=2)
        if self.swap:
            neg_dist_2 = func.pairwise_distance(positive, negative, p=2)
            neg_dist = torch.max(neg_dist, neg_dist_2)

        # triplet_loss = func.relu(pos_dist - neg_dist + self.margin)
        triplet_loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return triplet_loss.mean()


class TripletLoss_ratio(nn.Module):
    def __init__(self, swap=False, dist_func=None):
        super(TripletLoss_ratio, self).__init__()
        self.swap = swap
        if dist_func:
            self.dist_func = dist_func
        else:
            # self.dist_func = func.pairwise_distance
            self.dist_func = lambda x1, x2: torch.sqrt(func.pairwise_distance(x1, x2)+1e-8)
            
    def forward(self, anchor, positive, negative):
        pos_dist = self.dist_func(anchor, positive) # (N)
        neg_dist = self.dist_func(anchor, negative)
        if self.swap:
            neg_dist_2 = self.dist_func(positive, negative)
            neg_dist = torch.min(neg_dist, neg_dist_2)

        pos_dist = torch.clamp(pos_dist, max=10)
        neg_dist = torch.clamp(neg_dist, max=10)

        # ratio = pos_dist / (neg_dist + 1e-8)  
        # triplet_loss = torch.log1p(ratio).mean()

        ratio = torch.exp(pos_dist) / (torch.exp(pos_dist) + torch.exp(neg_dist))

        triplet_loss = ratio.mean()

        return triplet_loss


class HardTripletLoss(nn.Module):
    def __init__(self, margin=10.0):
        super(HardTripletLoss, self).__init__()
        self.margin = margin

    def hardest_triplet_selector(self, anchor, positive, negative):
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)
        # Select hardest positives (biggest positive distances)
        hardest_positive_dist = torch.min(pos_dist, dim=1)[0]
        # Select hardest negatives (smallest negative distances)
        hardest_negative_dist = torch.max(neg_dist, dim=1)[0]
        
        return hardest_positive_dist, hardest_negative_dist

    def forward(self, anchor, positive, negative):
        hardest_positive_dist, hardest_negative_dist = self.hardest_triplet_selector(anchor, positive, negative)
        triplet_loss = torch.clamp(hardest_positive_dist - hardest_negative_dist + self.margin, min=0.0)
        return triplet_loss.mean()
    
    
class HardTripletLoss_ratio(nn.Module):
    def __init__(self):
        super(HardTripletLoss_ratio, self).__init__()

    def hardest_triplet_selector(self, anchor, positive, negative):
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)
        # Select hardest positives (smallest positive distances)
        hardest_positive_dist = torch.min(pos_dist, dim=1)[0]
        # Select hardest negatives (smallest negative distances)
        hardest_negative_dist = torch.max(neg_dist, dim=1)[0]
        
        return hardest_positive_dist, hardest_negative_dist

    def forward(self, anchor, positive, negative):
        hardest_positive_dist, hardest_negative_dist = self.hardest_triplet_selector(anchor, positive, negative)
        ratio = hardest_positive_dist / (hardest_negative_dist + 1e-8) 
        triplet_loss = torch.log1p(ratio).mean()
        return triplet_loss