
import torch
import math
import functools
from torch.optim.lr_scheduler import _LRScheduler


def poly_learning_rate(epoch, warm_up_epoch, start_decay_epoch, total_epoch, min_lr):
    # Linear Warmup
    if (epoch < warm_up_epoch):
        return max(0, epoch / warm_up_epoch)
    else :
        lr = 1.0 - max(0, epoch - start_decay_epoch) /(float(total_epoch) - start_decay_epoch)

        if lr <= min_lr:
            lr = min_lr

    return lr

# ref: https://gaussian37.github.io/dl-pytorch-lr_scheduler/
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr




class CosineAnnealingWarmUpRestarts_grace(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=10, gamma=1., last_epoch=-1, warm_up=50):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        self.warm_up = warm_up
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if (self.T_cur < self.warm_up) and (self.cycle==0):
            return [self.base_eta_max for _ in self.base_lrs]
        elif (self.T_cur < (self.T_up + self.warm_up)) and (self.cycle>0):
            return [(self.eta_max - base_lr)*(self.T_cur-self.warm_up) / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            if self.cycle==0:
                return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.warm_up) / (self.T_i))) / 2 for base_lr in self.base_lrs]
            else:
                return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up-self.warm_up) / (self.T_i - self.T_up))) / 2 for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= (self.T_i+self.warm_up):
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= (self.T_0 + self.warm_up):
                if self.T_mult == 1:
                    # self.T_cur = epoch % self.T_0
                    self.cycle = (epoch - self.warm_up) // self.T_0
                    self.T_cur = epoch - self.cycle * self.T_0
                    
                else:
                    ## need to be updated
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr




def create_scheduler(name, optimizer, args):
    if name == 'poly_lr':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=functools.partial(poly_learning_rate, warm_up_epoch=args.warmup_epochs, start_decay_epoch=args.epochs/10, total_epoch=args.epochs, min_lr=args.min_lr))            

    elif name == 'cosine_annealing_warm_restart':
        # ref: https://gaussian37.github.io/dl-pytorch-lr_scheduler/
        lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=150, T_mult=1, eta_max=0.001, T_up=50, gamma=1)   ## ablation study 148

    elif name == 'cosine_annealing_warm_restart_grace':
        lr_scheduler = CosineAnnealingWarmUpRestarts_grace(optimizer, T_0=100, T_mult=1, eta_max=0.001, T_up=10, gamma=0.8, warm_up=100)
    
    elif name == 'reduce_LR_on_plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=10)

    else :
        raise KeyError("Wrong scheduler name `{}`".format(name))        


    return lr_scheduler