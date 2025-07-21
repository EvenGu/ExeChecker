import numpy as np
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':.4f', reset_frequency=1000):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.reset_frequency = reset_frequency

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if np.isnan(val):
            return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = ('{name} {val' + self.fmt + '} ({avg') + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



def accuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum()*1.0/dista.size()[0]
