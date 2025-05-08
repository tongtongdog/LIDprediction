from typing import Optional, List

import torch
from torch import nn
import torch.nn.functional as F

from utils import to_tensor
from torch.nn.modules.loss import _Loss



def soft_dice_score(output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None,) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)    ## elementwise multiplication
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score


class DiceLoss(_Loss):
    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
    ):
        """Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases
        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)
        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """


        assert mode in {'binary', 'multilabel', 'multiclass'}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != 'binary', "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == 'multiclass':
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)      ## batch size
        num_classes = y_pred.size(1)    ## used for multiclass and multilabel cases
        dims = (0, 2)

        if self.mode == 'binary':
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == 'multiclass':
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # H, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # H, C, H*W

        if self.mode == 'multilabel':
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = self.compute_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss
        
        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)       

        if self.classes is not None:
            loss = loss[self.classes]

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        ## probably needed because we are running in batches
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return soft_dice_score(output, target, smooth, eps, dims)



class Dice_BCE_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function_1 = DiceLoss(mode='binary', from_logits=True)
        self.loss_function_2 = torch.nn.BCEWithLogitsLoss()
        self.dice_weight     = 1.0   
        self.bce_weight      = 1.0   

    def forward(self, y_pred, y_true):
        dice_loss  = self.loss_function_1(y_pred, y_true)
        bce_loss   = self.loss_function_2(y_pred, y_true)

        return self.dice_weight*dice_loss + self.bce_weight*bce_loss




## Uptask Loss
class Uptask_Loss(torch.nn.Module):
    def __init__(self, name='Up_SMART_Net', epoch=0):
        super().__init__()

        weights = torch.tensor([1.0, 1.6], dtype=torch.float).cuda()   ## used for 402 PET images

        self.loss_cls     = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.3)
        self.loss_seg     = Dice_BCE_Loss()
        self.loss_rec     = torch.nn.L1Loss()
        
        self.name           = name
        self.epoch = epoch

        self.cls_weight     = 2.0
        self.seg_weight     = 1.0
        self.rec_weight     = 1.0


    def forward(self, cls_pred=None, seg_pred=None, rec_pred=None, cls_gt=None, seg_gt=None, rec_gt=None):
        if self.name == 'Up_SMART_Net':
            loss_cls     = self.loss_cls(cls_pred, cls_gt)
            loss_seg     = self.loss_seg(seg_pred, seg_gt)
            loss_rec     = self.loss_rec(rec_pred, rec_gt)
            
            total = self.cls_weight*loss_cls + self.seg_weight*loss_seg + self.rec_weight*loss_rec

            return total, {'CLS_Loss':(self.cls_weight*loss_cls).item(), 'SEG_Loss':(self.seg_weight*loss_seg).item(), 'REC_Loss':(self.rec_weight*loss_rec).item()}
        
        elif self.name == 'Up_SMART_Net_Dual_CLS_SEG':
            loss_cls     = self.loss_cls(cls_pred, cls_gt)
            loss_seg     = self.loss_seg(seg_pred, seg_gt)
            total        = self.cls_weight*loss_cls + self.seg_weight*loss_seg
            return total, {'CLS_Loss':(self.cls_weight*loss_cls).item(), 'SEG_Loss':(self.seg_weight*loss_seg).item()}

        elif self.name == 'Up_SMART_Net_Dual_CLS_REC':
            loss_cls     = self.loss_cls(cls_pred, cls_gt)
            loss_rec     = self.loss_rec(rec_pred, rec_gt)
            total        = self.cls_weight*loss_cls + self.rec_weight*loss_rec
            return total, {'CLS_Loss':(self.cls_weight*loss_cls).item(), 'REC_Loss':(self.rec_weight*loss_rec).item()}
        
        elif self.name == 'Up_SMART_Net_Dual_SEG_REC':
            loss_seg     = self.loss_seg(seg_pred, seg_gt)
            loss_rec     = self.loss_rec(rec_pred, rec_gt)
            total        = self.seg_weight*loss_seg + self.rec_weight*loss_rec
            return total, {'SEG_Loss':(self.seg_weight*loss_seg).item(), 'REC_Loss':(self.rec_weight*loss_rec).item()}

        elif self.name == 'Up_SMART_Net_Single_CLS':
            loss_cls     = self.loss_cls(cls_pred, cls_gt)
            total        = self.cls_weight*loss_cls 
            return total, {'CLS_Loss':(self.cls_weight*loss_cls).item()}

        else :
            raise KeyError("Wrong Loss name `{}`".format(self.name))  
