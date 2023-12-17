from torch.optim.lr_scheduler import _LRScheduler
import math 



class WarmupCosineAnnealingLRWithMinLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, lr_min=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_min = lr_min
        self.last_epoch = last_epoch
        super(WarmupCosineAnnealingLRWithMinLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.lr_min + (base_lr - self.lr_min) * self.last_epoch / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            return [
                self.lr_min + (base_lr - self.lr_min) * 0.5 * (1.0 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)))
                for base_lr in self.base_lrs
            ]