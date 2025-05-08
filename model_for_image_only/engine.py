import math
import utils
import numpy as np
import torch
import os

from metrics import *
import torch.nn.functional as F
import SimpleITK as sitk
from sklearn.metrics import roc_curve, auc

import scipy.ndimage as ndimage
from matplotlib import pyplot as plt
from skimage.transform import resize

from monai.transforms import SaveImage
from monai.transforms import Resize


def freeze_params(model: torch.nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False

def unfreeze_params(model: torch.nn.Module):
    """Set requires_grad=True for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = True

def predict(self, x):
    """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

    Args:
        x: 4D torch tensor with shape (batch_size, channels, height, width)

    Return:
        prediction: 4D torch tensor with shape (batch_size, classes, height, width)

    """
    if self.training:
        self.eval()

    with torch.no_grad():
        x = self.forward(x)

    return x


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def Activation_Map(x):

    print('Shape of X: ', x.shape)
    mean = torch.mean(x, dim=1)
    mean = torch.sigmoid(mean).squeeze(0).cpu().detach().numpy()  ## only the batch dimension removed

    mean = resize(mean, (64,96,64))
    print('Shape of mean (after resize): ', mean.shape)
    
    return mean


## Train
def train_Up_SMART_Net(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size):

    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)      # (B, C, H, W, D)
        seg_gt  = batch_data["seg_label"].to(device)      # (B, C, H, W, D)
        cls_gt  = batch_data["cls_label"].to(device)

        cls_pred, seg_pred, rec_pred = model(inputs)

        ## cls_pred & cls_gt : torch.Size([2, 2])
        ## seg_pred & seg_gt : torch.Size([2, 1, 192, 224, 112])
        ## inputs & rec_red : torch.Size([2, 1, 192, 224, 112])

        loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, rec_pred=rec_pred, cls_gt=cls_gt, seg_gt=seg_gt, rec_gt=inputs)
        loss_value = loss.item()


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print("image == ", batch_data['image_meta_dict']['filename_or_obj'])
            print("label == ", batch_data['label_meta_dict']['filename_or_obj'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
 
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}




## Valid
@torch.no_grad()
def valid_Up_SMART_Net(model, criterion, data_loader, device, epoch, print_freq, batch_size, output_dir_2):

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    batch_num, correct = 0, 0
    size = len(data_loader.dataset)
    y_true, y_prob, y_prob_bi = [], [], []
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)      # (B, C, H, W, D)
        seg_gt  = batch_data["seg_label"].to(device)      # (1, 1, H, W, D)
        cls_gt  = batch_data["cls_label"].to(device)      # (1, 2)

        cls_pred, seg_pred, rec_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, rec_pred=rec_pred, cls_gt=cls_gt, seg_gt=seg_gt, rec_gt=inputs)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value) 
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        cls_pred_prob = torch.sigmoid(cls_pred)
        seg_pred_prob = torch.sigmoid(seg_pred)


        cls_pred_result = cls_pred_prob.argmax(dim=1, keepdim=True)
        cls_pred_result = cls_pred_result.squeeze().detach().cpu().numpy()    # (1, 2) --> (2)
        cls_gt_result = cls_gt.argmax(dim=1, keepdim=True)
        cls_gt_result = cls_gt_result.squeeze().detach().cpu().numpy()    # (1, 2) --> (2)

        cls_pred_numpy = cls_pred.detach().cpu().numpy().astype('float32')
        cls_pred_prob_numpy = cls_pred_prob.detach().cpu().numpy().astype('float32')

        cls_log =  open(os.path.join(output_dir_2,'cls_pred.txt'), 'a')
        data = f'EPOCH {epoch} : PRED:  {cls_pred_result}, TRUE:  {cls_gt_result}, LOGIT:  {cls_pred_numpy}, PROBABILITY:  {cls_pred_prob_numpy}'
        cls_log.write(data + "\n")
        cls_log.close()

        if batch_num%40 == 0:

            ######## segmentation prediction save ########
            seg_pred_binarized = seg_pred_prob.round().squeeze().detach().cpu().numpy().astype('float32')    # (H, W, D)
            seg_pred_binarized = sitk.GetImageFromArray(seg_pred_binarized)
            sitk.WriteImage(seg_pred_binarized, os.path.join(output_dir_2, f'seg_pred_{epoch}_{batch_num}.nii.gz'))

            ######## reconstruction prediction save ######
            rec_pred_array = rec_pred.squeeze().detach().cpu().numpy().astype('float32')    # (H, W, D)
            rec_pred_array = sitk.GetImageFromArray(rec_pred_array)
            sitk.WriteImage(rec_pred_array, os.path.join(output_dir_2, f'rec_pred_{epoch}_{batch_num}.nii.gz'))

        batch_num += 1

        ## For Accuracy, AUROC calculation
        correct += (cls_pred_result == cls_gt_result).sum().item()
        y_true.append(np.float32(cls_gt_result))
        y_prob.append(cls_pred_prob_numpy[:,1][0])
        y_prob_bi.append(np.float32(cls_pred_result))

        # Metric CLS
        auc_1            = auc_metric(y_pred=cls_pred_prob, y=cls_gt)

        cls_pred_prob_one_hot = torch.argmax(cls_pred_prob, dim=1)
        cls_pred_prob_one_hot = torch.nn.functional.one_hot(cls_pred_prob_one_hot, num_classes=2)
        
        confuse_matrix = confuse_metric(y_pred=cls_pred_prob_one_hot, y=cls_gt)
        
        # Metrics SEG
        result_dice    = dice_metric(y_pred=seg_pred_prob.round(), y=seg_gt)

        # Metrics REC
        mae = torch.nn.functional.l1_loss(input=rec_pred, target=inputs).item()
        metric_logger.update(mae=mae)

 
    ## For Accuracy, AUROC calculation
    correct /= size

    tp, tn, fp, fn = 0, 0, 0, 0
    sensitivity, specificity, precision, f1_score = 0, 0, 0, 0
    for i in range(size):
        if y_true[i] == 0 and y_prob_bi[i] == 0:    
            tn +=1
        elif y_true[i] == 0 and y_prob_bi[i] == 1:
            fp += 1
        elif y_true[i] == 1 and y_prob_bi[i] == 0:
            fn += 1
        elif y_true[i] == 1 and y_prob_bi[i] == 1:
            tp += 1
        else:
            raise
    
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)

    ## precision denominator can equal 0 if all predicted values are negative!
    if tp+fp == 0:
        precision = -1
    else: 
        precision = tp/(tp+fp)
    
    f1_score = 2*tp/(2*tp+fp+fn)


    # Aggregatation
    auc_1                = auc_metric.aggregate()
    dice               = dice_metric.aggregate().item()   

    fpr, tpr, th = roc_curve(y_true, y_prob)
    auc_2 = auc(fpr, tpr)

    metric_logger.update(auc=auc_1)
    metric_logger.update(auc_2=auc_2)          
    metric_logger.update(acc_2=correct, sen_2=sensitivity, spe_2=specificity, pre_2=precision, f1_score_2=f1_score)
    metric_logger.update(dice=dice)
        
    auc_metric.reset()
    confuse_metric.reset()
    dice_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}



## Test 
@torch.no_grad()
def test_Up_SMART_Net(model, criterion, data_loader, device, print_freq, batch_size, output_dir, idx=0):

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'

    batch_num, correct = 0, 0
    size = len(data_loader.dataset)
    y_true, y_prob, y_prob_bi = [], [], []
    cls_pred_prob_list, cls_gt_list= [], []

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)      # (B, C, H, W, D)
        seg_gt  = batch_data["seg_label"].to(device)      # (B, C, H, W, D)
        cls_gt  = batch_data["cls_label"].to(device)

        cls_pred, seg_pred, rec_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, rec_pred=rec_pred, cls_gt=cls_gt, seg_gt=seg_gt, rec_gt=inputs)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value) 
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        cls_pred_prob = torch.sigmoid(cls_pred)
        seg_pred_prob = torch.sigmoid(seg_pred)


        cls_pred_result = cls_pred_prob.argmax(dim=1, keepdim=True)
        cls_pred_result = cls_pred_result.squeeze().detach().cpu().numpy()    # (1, 2) --> (2)
        cls_gt_result = cls_gt.argmax(dim=1, keepdim=True)
        cls_gt_list.append(cls_gt_result)  ## for k-fold cross validation

        cls_gt_result = cls_gt_result.squeeze().detach().cpu().numpy()    # (1, 2) --> (2)

        cls_pred_numpy = cls_pred.detach().cpu().numpy().astype('float32')
        cls_pred_prob_list.append(cls_pred_prob[0])   ## for k-fold cross validation
        
        cls_pred_prob_numpy = cls_pred_prob.detach().cpu().numpy().astype('float32')

        cls_log =  open(os.path.join(output_dir,f'cls_pred_{idx}.txt'), 'a')
        data = f'PRED:  {cls_pred_result}, TRUE:  {cls_gt_result}, LOGIT:  {cls_pred_numpy}, PROBABILITY:  {cls_pred_prob_numpy}'
        cls_log.write(data + "\n")
        cls_log.close()


        ## For Accuracy, AUROC calculation
        correct += (cls_pred_result == cls_gt_result).sum().item()
        y_true.append(np.float32(cls_gt_result))
        y_prob.append(cls_pred_prob_numpy[:,1][0])
        y_prob_bi.append(np.float32(cls_pred_result))

        # Metric CLS
        auc_1            = auc_metric(y_pred=cls_pred_prob, y=cls_gt)

        cls_pred_prob_one_hot = torch.argmax(cls_pred_prob, dim=1)
        cls_pred_prob_one_hot = torch.nn.functional.one_hot(cls_pred_prob_one_hot, num_classes=2)
        
        confuse_matrix = confuse_metric(y_pred=cls_pred_prob_one_hot, y=cls_gt)
        
        # Metrics SEG
        result_dice    = dice_metric(y_pred=seg_pred_prob.round(), y=seg_gt)

        # Metrics REC
        mae = torch.nn.functional.l1_loss(input=rec_pred, target=inputs).item()
        metric_logger.update(mae=mae)

 
    ## For Accuracy, AUROC calculation 
    correct /= size

    tp, tn, fp, fn = 0, 0, 0, 0
    sensitivity, specificity, precision, f1_score = 0, 0, 0, 0
    for i in range(size):
        if y_true[i] == 0 and y_prob_bi[i] == 0:    
            tn +=1
        elif y_true[i] == 0 and y_prob_bi[i] == 1:
            fp += 1
        elif y_true[i] == 1 and y_prob_bi[i] == 0:
            fn += 1
        elif y_true[i] == 1 and y_prob_bi[i] == 1:
            tp += 1
        else:
            raise
    
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)

    ## precision denominator can equal 0 if all predicted values are negative!
    if tp+fp == 0:
        precision = -1
    else: 
        precision = tp/(tp+fp)
    
    f1_score = 2*tp/(2*tp+fp+fn)


    # Aggregatation
    auc_1                = auc_metric.aggregate()
    dice               = dice_metric.aggregate().item()   

    fpr, tpr, th = roc_curve(y_true, y_prob)
    auc_2 = auc(fpr, tpr)

    metric_logger.update(auc=auc_1)
    metric_logger.update(auc_2=auc_2)          
    metric_logger.update(acc_2=correct, sen_2=sensitivity, spe_2=specificity, pre_2=precision, f1_score_2=f1_score)
    metric_logger.update(dice=dice)
        
    auc_metric.reset()
    confuse_metric.reset()
    dice_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}, cls_pred_prob_list, cls_gt_list



## Inference code 
@torch.no_grad()
def infer_Up_SMART_Net(model, data_loader, device, print_freq, save_dir):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)
    header = 'TEST:'
    
    save_dict = dict()
    img_path_list = []
    img_list = []
    cls_list = []
    seg_list = []
    rec_list = []
    act_list = []

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)      # (B, C, H, W, D)
        file_name = os.path.basename(batch_data['image_meta_dict']['filename_or_obj'][0])[:-7]

        # register forward hook
        model.module.encoders[-2].basic_module.non_linearity.register_forward_hook(get_activation('Activation Map')) # for Activation Map

        cls_pred, seg_pred, rec_pred = model(inputs)

        # post-processing
        cls_pred = torch.sigmoid(cls_pred)
        seg_pred = torch.sigmoid(seg_pred)

        current_act_map = Activation_Map(activation['Activation Map'])

        img_path_list.append(batch_data['image_meta_dict']['filename_or_obj'][0])
        img_list.append(inputs.detach().cpu().squeeze())
        cls_list.append(cls_pred.detach().cpu().squeeze())
        seg_list.append(seg_pred.detach().cpu().squeeze())
        rec_list.append(rec_pred.detach().cpu().squeeze())
        act_list.append(current_act_map)

        sitk_img = sitk.GetImageFromArray(current_act_map)
        sitk.WriteImage(sitk_img, os.path.join(save_dir, f'{file_name}_activation_map.nii.gz'))


    save_dict['img_path_list']  = img_path_list
    save_dict['img_list']       = img_list
    save_dict['cls_pred']       = cls_list
    save_dict['seg_pred']       = seg_list
    save_dict['rec_pred']       = rec_list
    save_dict['activation_map'] = act_list
    np.savez(os.path.join(save_dir, f'{file_name}_result.npz'), result=save_dict) 






## DUAL

############## CLS+SEG ##################
## Train
def train_Up_SMART_Net_Dual_CLS_SEG(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size):
    # 2d slice-wise based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)      # (B, C, H, W, D)
        seg_gt  = batch_data["seg_label"].to(device)      # (B, C, H, W, D)
        cls_gt  = batch_data["cls_label"].to(device)

        cls_pred, seg_pred = model(inputs)
        loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, cls_gt=cls_gt, seg_gt=seg_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print("image == ", batch_data['image_meta_dict']['filename_or_obj'])
            print("label == ", batch_data['label_meta_dict']['filename_or_obj'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
 
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

## Valid
@torch.no_grad()
def valid_Up_SMART_Net_Dual_CLS_SEG(model, criterion, data_loader, device, epoch, print_freq, batch_size, output_dir_2):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    batch_num, correct = 0, 0
    size = len(data_loader.dataset)
    y_true, y_prob, y_prob_bi = [], [], []

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)      # (B, C, H, W, D)
        seg_gt  = batch_data["seg_label"].to(device)      # (1, 1, H, W, D)
        cls_gt  = batch_data["cls_label"].to(device)      # (1, 2)

        cls_pred, seg_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, cls_gt=cls_gt, seg_gt=seg_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value) 
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        cls_pred_prob = torch.sigmoid(cls_pred)
        seg_pred_prob = torch.sigmoid(seg_pred)

        cls_pred_result = cls_pred_prob.round().argmax(dim=1, keepdim=True)
        cls_pred_result = cls_pred_result.squeeze().detach().cpu().numpy()    # (1, 2) --> (2)
        cls_gt_result = cls_gt.argmax(dim=1, keepdim=True)
        cls_gt_result = cls_gt_result.squeeze().detach().cpu().numpy()    # (1, 2) --> (2)

        cls_pred_numpy = cls_pred.detach().cpu().numpy().astype('float32')
        cls_pred_prob_numpy = cls_pred_prob.detach().cpu().numpy().astype('float32')

        cls_log =  open(os.path.join(output_dir_2,'cls_pred.txt'), 'a')
        data = f'EPOCH {epoch} : PRED:  {cls_pred_result}, TRUE:  {cls_gt_result}, LOGIT:  {cls_pred_numpy}, PROBABILITY:  {cls_pred_prob_numpy}'
        cls_log.write(data + "\n")
        cls_log.close()

        if batch_num%40 == 0:
            ######## segmentation prediction save ########
            seg_pred_binarized = seg_pred_prob.round().squeeze().cpu().numpy().astype('float32')    # (H, W, D)
            seg_pred_binarized = sitk.GetImageFromArray(seg_pred_binarized)
            sitk.WriteImage(seg_pred_binarized, os.path.join(output_dir_2, f'seg_pred_{epoch}_{batch_num}.nii.gz'))

        batch_num += 1

        ## For Accuracy, AUROC calculation
        correct += (cls_pred_result == cls_gt_result).sum().item()
        y_true.append(np.float32(cls_gt_result))
        y_prob.append(cls_pred_prob_numpy[:,1][0])
        y_prob_bi.append(np.float32(cls_pred_result))

        # Metric CLS
        auc_1            = auc_metric(y_pred=cls_pred_prob, y=cls_gt)

        cls_pred_prob_one_hot = torch.argmax(cls_pred_prob, dim=1)
        cls_pred_prob_one_hot = torch.nn.functional.one_hot(cls_pred_prob_one_hot, num_classes=2)

        confuse_matrix = confuse_metric(y_pred=cls_pred_prob.round(), y=cls_gt)   # pred_cls must be round() !!

        # Metrics SEG
        result_dice = dice_metric(y_pred=seg_pred_prob.round(), y=seg_gt)              # pred_seg must be round() !! 

    ## For Accuracy, AUROC calculation
    correct /= size

    tp, tn, fp, fn = 0, 0, 0, 0
    sensitivity, specificity, precision, f1_score = 0, 0, 0, 0
    for i in range(size):
        if y_true[i] == 0 and y_prob_bi[i] == 0:    
            tn +=1
        elif y_true[i] == 0 and y_prob_bi[i] == 1:
            fp += 1
        elif y_true[i] == 1 and y_prob_bi[i] == 0:
            fn += 1
        elif y_true[i] == 1 and y_prob_bi[i] == 1:
            tp += 1
        else:
            raise
    
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)

    ## precision denominator can equal 0 if all predicted values are negative!
    if tp+fp == 0:
        precision = -1
    else: 
        precision = tp/(tp+fp)
    
    f1_score = 2*tp/(2*tp+fp+fn)

    
    # Aggregatation
    auc_1                = auc_metric.aggregate()
    dice               = dice_metric.aggregate().item()    
    
    fpr, tpr, th = roc_curve(y_true, y_prob)
    auc_2 = auc(fpr, tpr)

    metric_logger.update(auc=auc_1)
    metric_logger.update(auc_2=auc_2)          
    metric_logger.update(acc_2=correct, sen_2=sensitivity, spe_2=specificity, pre_2=precision, f1_score_2=f1_score)
    metric_logger.update(dice=dice)
    
    auc_metric.reset()
    confuse_metric.reset()
    dice_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


## Test    
@torch.no_grad()
def test_Up_SMART_Net_Dual_CLS_SEG(model, criterion, data_loader, device, print_freq, batch_size, output_dir, idx=0):

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'

    batch_num, correct = 0, 0
    size = len(data_loader.dataset)
    y_true, y_prob, y_prob_bi = [], [], []
    cls_pred_prob_list, cls_gt_list= [], []

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)      # (B, C, H, W, D)
        seg_gt  = batch_data["seg_label"].to(device)      # (B, C, H, W, D)
        cls_gt  = batch_data["cls_label"].to(device)

        cls_pred, seg_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, cls_gt=cls_gt, seg_gt=seg_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        cls_pred_prob = torch.sigmoid(cls_pred)
        seg_pred_prob = torch.sigmoid(seg_pred)

        cls_pred_result = cls_pred_prob.argmax(dim=1, keepdim=True)
        cls_pred_result = cls_pred_result.squeeze().detach().cpu().numpy()    # (1, 2) --> (2)
        cls_gt_result = cls_gt.argmax(dim=1, keepdim=True)
        cls_gt_list.append(cls_gt_result)  ## for k-fold cross validation

        cls_gt_result = cls_gt_result.squeeze().detach().cpu().numpy()    # (1, 2) --> (2)

        cls_pred_numpy = cls_pred.detach().cpu().numpy().astype('float32')
        cls_pred_prob_list.append(cls_pred_prob[0])   ## for k-fold cross validation
        
        cls_pred_prob_numpy = cls_pred_prob.detach().cpu().numpy().astype('float32')

        cls_log =  open(os.path.join(output_dir,f'cls_pred_{idx}.txt'), 'a')
        data = f'PRED:  {cls_pred_result}, TRUE:  {cls_gt_result}, LOGIT:  {cls_pred_numpy}, PROBABILITY:  {cls_pred_prob_numpy}'
        cls_log.write(data + "\n")
        cls_log.close()


        ## For Accuracy, AUROC calculation
        correct += (cls_pred_result == cls_gt_result).sum().item()
        y_true.append(np.float32(cls_gt_result))
        y_prob.append(cls_pred_prob_numpy[:,1][0])
        y_prob_bi.append(np.float32(cls_pred_result))

        # Metric CLS
        auc_1            = auc_metric(y_pred=cls_pred_prob, y=cls_gt)

        cls_pred_prob_one_hot = torch.argmax(cls_pred_prob, dim=1)
        cls_pred_prob_one_hot = torch.nn.functional.one_hot(cls_pred_prob_one_hot, num_classes=2)
        
        confuse_matrix = confuse_metric(y_pred=cls_pred_prob_one_hot, y=cls_gt)
        
        # Metrics SEG
        result_dice    = dice_metric(y_pred=seg_pred_prob.round(), y=seg_gt)


    ## For Accuracy, AUROC calculation
    correct /= size

    tp, tn, fp, fn = 0, 0, 0, 0
    sensitivity, specificity, precision, f1_score = 0, 0, 0, 0
    for i in range(size):
        if y_true[i] == 0 and y_prob_bi[i] == 0:    
            tn +=1
        elif y_true[i] == 0 and y_prob_bi[i] == 1:
            fp += 1
        elif y_true[i] == 1 and y_prob_bi[i] == 0:
            fn += 1
        elif y_true[i] == 1 and y_prob_bi[i] == 1:
            tp += 1
        else:
            raise
    
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)

    ## precision denominator can equal 0 if all predicted values are negative!
    if tp+fp == 0:
        precision = -1
    else: 
        precision = tp/(tp+fp)
    
    f1_score = 2*tp/(2*tp+fp+fn)


    # Aggregatation
    auc_1                = auc_metric.aggregate()
    dice               = dice_metric.aggregate().item()   

    fpr, tpr, th = roc_curve(y_true, y_prob)
    auc_2 = auc(fpr, tpr)

    metric_logger.update(auc=auc_1)
    metric_logger.update(auc_2=auc_2)          
    metric_logger.update(acc_2=correct, sen_2=sensitivity, spe_2=specificity, pre_2=precision, f1_score_2=f1_score)
    metric_logger.update(dice=dice)
        
    auc_metric.reset()
    confuse_metric.reset()
    dice_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}, cls_pred_prob_list, cls_gt_list
    




################ CLS+REC #######################
## Train
def train_Up_SMART_Net_Dual_CLS_REC(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size, freeze_feature_extractor):

    print('freeze_feature_extractor set as: ', freeze_feature_extractor)

    ####### added for freezing #############
    if freeze_feature_extractor:
        # Freeze the feature extractor part of the model
        freeze_params(model.module.encoders)    
        # freeze_params(model.module.seg_decoder)    
        # freeze_params(model.module.seg_final_conv)    
        freeze_params(model.module.rec_decoder)
        freeze_params(model.module.rec_final_conv)  
    else:
        print('####### No Freezing --> Dobby is FREE!! ########')  
    #######################################

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Updated Number of Learnable Params:', n_parameters) 


    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)      # (B, C, H, W, D)
        cls_gt  = batch_data["cls_label"].to(device)

        cls_pred, rec_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, rec_pred=rec_pred, cls_gt=cls_gt, rec_gt=inputs)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print("image == ", batch_data['image_meta_dict']['filename_or_obj'])
            print("label == ", batch_data['label_meta_dict']['filename_or_obj'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
 
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

## Valid
@torch.no_grad()
def valid_Up_SMART_Net_Dual_CLS_REC(model, criterion, data_loader, device, epoch, print_freq, batch_size, output_dir_2):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    batch_num, correct = 0, 0
    size = len(data_loader.dataset)
    y_true, y_prob, y_prob_bi = [], [], []

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)      # (B, C, H, W, D)
        seg_gt  = batch_data["seg_label"].to(device)      # (1, 1, H, W, D)
        cls_gt  = batch_data["cls_label"].to(device)      # (1, 2)

        cls_pred, rec_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, rec_pred=rec_pred, cls_gt=cls_gt, rec_gt=inputs)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value) 
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        cls_pred_prob = torch.sigmoid(cls_pred)

        ######## classification prediction save #############################
        cls_pred_result = cls_pred_prob.round().argmax(dim=1, keepdim=True)
        cls_pred_result = cls_pred_result.squeeze().detach().cpu().numpy()    # (1, 2) --> (2)
        cls_gt_result = cls_gt.argmax(dim=1, keepdim=True)
        cls_gt_result = cls_gt_result.squeeze().detach().cpu().numpy()    # (1, 2) --> (2)

        cls_pred_numpy = cls_pred.detach().cpu().numpy().astype('float32')
        cls_pred_prob_numpy = cls_pred_prob.detach().cpu().numpy().astype('float32')

        cls_log =  open(os.path.join(output_dir_2,'cls_pred.txt'), 'a')
        data = f'EPOCH {epoch} : PRED:  {cls_pred_result}, TRUE:  {cls_gt_result}, LOGIT:  {cls_pred_numpy}, PROBABILITY:  {cls_pred_prob_numpy}'
        cls_log.write(data + "\n")
        cls_log.close()

        if batch_num%40 == 0:
            ######## reconstruction prediction save ######
            rec_pred_array = rec_pred.squeeze().cpu().numpy().astype('float32')    # (H, W, D)
            rec_pred_array = sitk.GetImageFromArray(rec_pred_array)
            sitk.WriteImage(rec_pred_array, os.path.join(output_dir_2, f'rec_pred_{epoch}_{batch_num}.nii.gz'))
            ##############################################

        batch_num += 1
        #########################################################################

        ## For Accuracy, AUROC calculation 
        correct += (cls_pred_result == cls_gt_result).sum().item()
        y_true.append(np.float32(cls_gt_result))
        y_prob.append(cls_pred_prob_numpy[:,1][0])
        y_prob_bi.append(np.float32(cls_pred_result))

        # Metric CLS
        auc_1            = auc_metric(y_pred=cls_pred_prob, y=cls_gt)

        cls_pred_prob_one_hot = torch.argmax(cls_pred_prob, dim=1)
        cls_pred_prob_one_hot = torch.nn.functional.one_hot(cls_pred_prob_one_hot, num_classes=2)

        confuse_matrix = confuse_metric(y_pred=cls_pred_prob.round(), y=cls_gt)   # pred_cls must be round() !!
  
        # Metrics REC
        mae = torch.nn.functional.l1_loss(input=rec_pred, target=inputs).item()
        metric_logger.update(mae=mae)

    ## For Accuracy, AUROC calculation
    correct /= size

    tp, tn, fp, fn = 0, 0, 0, 0
    sensitivity, specificity, precision, f1_score = 0, 0, 0, 0
    for i in range(size):
        if y_true[i] == 0 and y_prob_bi[i] == 0:    
            tn +=1
        elif y_true[i] == 0 and y_prob_bi[i] == 1:
            fp += 1
        elif y_true[i] == 1 and y_prob_bi[i] == 0:
            fn += 1
        elif y_true[i] == 1 and y_prob_bi[i] == 1:
            tp += 1
        else:
            raise
    
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)

    ## precision denominator can equal 0 if all predicted values are negative!
    if tp+fp == 0:
        precision = -1
    else: 
        precision = tp/(tp+fp)
    
    f1_score = 2*tp/(2*tp+fp+fn)


    # Aggregatation
    auc_1                = auc_metric.aggregate()
        
    fpr, tpr, th = roc_curve(y_true, y_prob)
    auc_2 = auc(fpr, tpr)

    metric_logger.update(auc=auc_1)          
    metric_logger.update(auc_2=auc_2)          
    metric_logger.update(acc_2=correct, sen_2=sensitivity, spe_2=specificity, pre_2=precision, f1_score_2=f1_score)
    
    auc_metric.reset()
    confuse_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

## Test    
@torch.no_grad()
def test_Up_SMART_Net_Dual_CLS_REC(model, criterion, data_loader, device, print_freq, batch_size, output_dir, idx=0):

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'

    batch_num, correct = 0, 0
    size = len(data_loader.dataset)
    y_true, y_prob, y_prob_bi = [], [], []
    cls_pred_prob_list, cls_gt_list= [], []

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)      # (B, C, H, W, D)
        cls_gt  = batch_data["cls_label"].to(device)

        cls_pred, rec_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, rec_pred=rec_pred, cls_gt=cls_gt, rec_gt=inputs)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value) 
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        cls_pred_prob = torch.sigmoid(cls_pred)

        ######## classification prediction save #############################
        cls_pred_result = cls_pred_prob.argmax(dim=1, keepdim=True)
        cls_pred_result = cls_pred_result.squeeze().detach().cpu().numpy()    # (1, 2) --> (2)
        cls_gt_result = cls_gt.argmax(dim=1, keepdim=True)
        cls_gt_list.append(cls_gt_result)  ## for k-fold cross validation

        cls_gt_result = cls_gt_result.squeeze().detach().cpu().numpy()    # (1, 2) --> (2)

        cls_pred_numpy = cls_pred.detach().cpu().numpy().astype('float32')
        cls_pred_prob_list.append(cls_pred_prob[0])   ## for k-fold cross validation
        
        cls_pred_prob_numpy = cls_pred_prob.detach().cpu().numpy().astype('float32')

        cls_log =  open(os.path.join(output_dir,f'cls_pred_{idx}.txt'), 'a')
        data = f'PRED:  {cls_pred_result}, TRUE:  {cls_gt_result}, LOGIT:  {cls_pred_numpy}, PROBABILITY:  {cls_pred_prob_numpy}'
        cls_log.write(data + "\n")
        cls_log.close()


        ## For Accuracy, AUROC calculation 
        correct += (cls_pred_result == cls_gt_result).sum().item()
        y_true.append(np.float32(cls_gt_result))
        y_prob.append(cls_pred_prob_numpy[:,1][0])
        y_prob_bi.append(np.float32(cls_pred_result))

        # Metric CLS
        auc_1            = auc_metric(y_pred=cls_pred_prob, y=cls_gt)

        cls_pred_prob_one_hot = torch.argmax(cls_pred_prob, dim=1)
        cls_pred_prob_one_hot = torch.nn.functional.one_hot(cls_pred_prob_one_hot, num_classes=2)
        
        confuse_matrix = confuse_metric(y_pred=cls_pred_prob_one_hot, y=cls_gt)
        
        # Metrics REC
        mae = torch.nn.functional.l1_loss(input=rec_pred, target=inputs).item()
        metric_logger.update(mae=mae)

 
    ## For Accuracy, AUROC calculation a
    correct /= size

    tp, tn, fp, fn = 0, 0, 0, 0
    sensitivity, specificity, precision, f1_score = 0, 0, 0, 0
    for i in range(size):
        if y_true[i] == 0 and y_prob_bi[i] == 0:    
            tn +=1
        elif y_true[i] == 0 and y_prob_bi[i] == 1:
            fp += 1
        elif y_true[i] == 1 and y_prob_bi[i] == 0:
            fn += 1
        elif y_true[i] == 1 and y_prob_bi[i] == 1:
            tp += 1
        else:
            raise
    
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)

    ## precision denominator can equal 0 if all predicted values are negative!
    if tp+fp == 0:
        precision = -1
    else: 
        precision = tp/(tp+fp)
    
    f1_score = 2*tp/(2*tp+fp+fn)


    # Aggregatation
    auc_1                = auc_metric.aggregate()

    fpr, tpr, th = roc_curve(y_true, y_prob)
    auc_2 = auc(fpr, tpr)

    metric_logger.update(auc=auc_1)
    metric_logger.update(auc_2=auc_2)          
    metric_logger.update(acc_2=correct, sen_2=sensitivity, spe_2=specificity, pre_2=precision, f1_score_2=f1_score)
        
    auc_metric.reset()
    confuse_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}, cls_pred_prob_list, cls_gt_list



## Inference code 
@torch.no_grad()
def infer_Up_SMART_Net_Dual_CLS_REC(model, data_loader, device, print_freq, save_dir):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)
    header = 'TEST:'
    
    save_dict = dict()
    img_path_list = []
    img_list = []
    cls_list = []
    rec_list = []
    act_list = []

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)      # (B, C, H, W, D)
        file_name = os.path.basename(batch_data['image_meta_dict']['filename_or_obj'][0])[:-7]

        # register forward hook
        # model.module.encoders[-2].basic_module.non_linearity.register_forward_hook(get_activation('Activation Map')) # for Activation Map
        model.module.encoders[-1].basic_module.non_linearity.register_forward_hook(get_activation('Activation Map')) # for Activation Map

        cls_pred, rec_pred = model(inputs)

        # post-processing
        cls_pred = torch.sigmoid(cls_pred)

        current_act_map = Activation_Map(activation['Activation Map'])

        img_path_list.append(batch_data['image_meta_dict']['filename_or_obj'][0])
        img_list.append(inputs.detach().cpu().squeeze())
        cls_list.append(cls_pred.detach().cpu().squeeze())
        rec_list.append(rec_pred.detach().cpu().squeeze())
        act_list.append(current_act_map)

        sitk_img = sitk.GetImageFromArray(current_act_map)
        sitk.WriteImage(sitk_img, os.path.join(save_dir, f'{file_name}_activation_map.nii.gz'))


    save_dict['img_path_list']  = img_path_list
    save_dict['img_list']       = img_list
    save_dict['cls_pred']       = cls_list
    save_dict['rec_pred']       = rec_list
    save_dict['activation_map'] = act_list
    np.savez(os.path.join(save_dir, f'{file_name}_result.npz'), result=save_dict) 






#################### SEG+REC #########################
## Train
def train_Up_SMART_Net_Dual_SEG_REC(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size):
    # 2d slice-wise based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)      # (B, C, H, W, D)
        seg_gt  = batch_data["seg_label"].to(device)      # (B, C, H, W, D)

        seg_pred, rec_pred = model(inputs)

        loss, loss_detail = criterion(seg_pred=seg_pred, rec_pred=rec_pred, seg_gt=seg_gt, rec_gt=inputs)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
 
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}
##################################################################################################


## Valid 
@torch.no_grad()
def valid_Up_SMART_Net_Dual_SEG_REC(model, criterion, data_loader, device, epoch, print_freq, batch_size, output_dir_2):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    batch_num = 0
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)      # (B, C, H, W, D)
        seg_gt  = batch_data["seg_label"].to(device)      # (1, 1, H, W, D)

        seg_pred, rec_pred = model(inputs)

        loss, loss_detail = criterion(seg_pred=seg_pred, rec_pred=rec_pred, seg_gt=seg_gt, rec_gt=inputs)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        seg_pred_prob = torch.sigmoid(seg_pred)


        if batch_num%20 == 0:
            ######## segmentation prediction save ########
            seg_pred_binarized = seg_pred_prob.round().squeeze().cpu().numpy().astype('float32')    # (H, W, D)
            seg_pred_binarized = sitk.GetImageFromArray(seg_pred_binarized)
            sitk.WriteImage(seg_pred_binarized, os.path.join(output_dir_2, f'seg_pred_{epoch}_{batch_num}.nii.gz'))
            ###############################################

            ######## reconstruction prediction save ######
            rec_pred_array = rec_pred.squeeze().cpu().numpy().astype('float32')    # (H, W, D)
            rec_pred_array = sitk.GetImageFromArray(rec_pred_array)
            sitk.WriteImage(rec_pred_array, os.path.join(output_dir_2, f'rec_pred_{epoch}_{batch_num}.nii.gz'))
            ##############################################

        batch_num += 1

        # Metrics SEG
        result_dice = dice_metric(y_pred=seg_pred_prob.round(), y=seg_gt)              # pred_seg must be round() !! 
  
        # Metrics REC
        mae = torch.nn.functional.l1_loss(input=rec_pred, target=inputs).item()
        metric_logger.update(mae=mae)

    # Aggregatation
    dice               = dice_metric.aggregate().item()    
    metric_logger.update(dice=dice)
    dice_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}
##################################################################################################



## Test 
@torch.no_grad()
def test_Up_SMART_Net_Dual_SEG_REC(model, criterion, data_loader, device, print_freq, batch_size):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)      # (B, C, H, W, D)
        seg_gt  = batch_data["seg_label"].to(device)      # (B, C, H, W, D)

        seg_pred, rec_pred = model(inputs)

        loss, loss_detail = criterion(seg_pred=seg_pred, rec_pred=rec_pred, seg_gt=seg_gt, rec_gt=inputs)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        seg_pred = torch.sigmoid(seg_pred)

        # Metrics SEG
        dice = dice_metric(y_pred=seg_pred.round(), y=seg_gt)              # pred_seg must be round() !!
        metric_logger.update(dice=dice)
  
        # Metrics REC
        mae = torch.nn.functional.l1_loss(input=rec_pred, target=inputs).item()
        metric_logger.update(mae=mae)

    # Aggregatation
    dice               = dice_metric.aggregate().item()    
    metric_logger.update(dice=dice)
    dice_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}
##################################################################################################




## Single
############# CLS #####################

## Train
def train_Up_SMART_Net_Single_CLS(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size):
    # 2d slice-wise based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)      # (B, C, H, W, D)
        cls_gt  = batch_data["cls_label"].to(device)

        cls_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print("image == ", batch_data['image_meta_dict']['filename_or_obj'])
            print("label == ", batch_data['label_meta_dict']['filename_or_obj'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
 
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

## Valid
@torch.no_grad()
def valid_Up_SMART_Net_Single_CLS(model, criterion, data_loader, device, epoch, print_freq, batch_size, output_dir_2):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    batch_num, correct = 0, 0
    size = len(data_loader.dataset)
    y_true, y_prob, y_prob_bi = [], [], []

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)      # (B, C, H, W, D)
        cls_gt  = batch_data["cls_label"].to(device)      # (1, 2)

        cls_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value) 
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        cls_pred_prob = torch.sigmoid(cls_pred)

        ######## classification prediction save #############################
        cls_pred_result = cls_pred_prob.round().argmax(dim=1, keepdim=True)
        cls_pred_result = cls_pred_result.squeeze().detach().cpu().numpy()    # (1, 2) --> (2)
        cls_gt_result = cls_gt.argmax(dim=1, keepdim=True)
        cls_gt_result = cls_gt_result.squeeze().detach().cpu().numpy()    # (1, 2) --> (2)

        cls_pred_numpy = cls_pred.detach().cpu().numpy().astype('float32')
        cls_pred_prob_numpy = cls_pred_prob.detach().cpu().numpy().astype('float32')

        cls_log =  open(os.path.join(output_dir_2,'cls_pred.txt'), 'a')
        data = f'EPOCH {epoch} : PRED:  {cls_pred_result}, TRUE:  {cls_gt_result}, LOGIT:  {cls_pred_numpy}, PROBABILITY:  {cls_pred_prob_numpy}'
        cls_log.write(data + "\n")
        cls_log.close()

        batch_num += 1
        #########################################################################

        ## For Accuracy, AUROC calculation a
        correct += (cls_pred_result == cls_gt_result).sum().item()
        y_true.append(np.float32(cls_gt_result))
        y_prob.append(cls_pred_prob_numpy[:,1][0])
        y_prob_bi.append(np.float32(cls_pred_result))


        # Metric CLS
        auc_1            = auc_metric(y_pred=cls_pred_prob, y=cls_gt)

        cls_pred_prob_one_hot = torch.argmax(cls_pred_prob, dim=1)
        cls_pred_prob_one_hot = torch.nn.functional.one_hot(cls_pred_prob_one_hot, num_classes=2)

        confuse_matrix = confuse_metric(y_pred=cls_pred_prob.round(), y=cls_gt)   # pred_cls must be round() !!

    ## For Accuracy, AUROC calculation 
    correct /= size

    tp, tn, fp, fn = 0, 0, 0, 0
    sensitivity, specificity, precision, f1_score = 0, 0, 0, 0
    for i in range(size):
        if y_true[i] == 0 and y_prob_bi[i] == 0:    
            tn +=1
        elif y_true[i] == 0 and y_prob_bi[i] == 1:
            fp += 1
        elif y_true[i] == 1 and y_prob_bi[i] == 0:
            fn += 1
        elif y_true[i] == 1 and y_prob_bi[i] == 1:
            tp += 1
        else:
            raise
    
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)

    ## precision denominator can equal 0 if all predicted values are negative!
    if tp+fp == 0:
        precision = -1
    else: 
        precision = tp/(tp+fp)
    
    f1_score = 2*tp/(2*tp+fp+fn)


    # Aggregatation
    auc_1                = auc_metric.aggregate()
        
    fpr, tpr, th = roc_curve(y_true, y_prob)
    auc_2 = auc(fpr, tpr)

    metric_logger.update(auc=auc_1)          
    metric_logger.update(auc_2=auc_2)          
    metric_logger.update(acc_2=correct, sen_2=sensitivity, spe_2=specificity, pre_2=precision, f1_score_2=f1_score)
    
    auc_metric.reset()
    confuse_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}



## Test    
@torch.no_grad()
def test_Up_SMART_Net_Single_CLS(model, criterion, data_loader, device, print_freq, batch_size, output_dir, idx=0):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'

    batch_num, correct = 0, 0
    size = len(data_loader.dataset)
    y_true, y_prob, y_prob_bi = [], [], []
    cls_pred_prob_list, cls_gt_list= [], []

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)      # (B, C, H, W, D)
        seg_gt  = batch_data["seg_label"].to(device)      # (B, C, H, W, D)
        cls_gt  = batch_data["cls_label"].to(device)

        cls_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value) 
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        cls_pred_prob = torch.sigmoid(cls_pred)

        ######## classification prediction save #############################
        cls_pred_result = cls_pred_prob.argmax(dim=1, keepdim=True)
        cls_pred_result = cls_pred_result.squeeze().detach().cpu().numpy()    # (1, 2) --> (2)
        cls_gt_result = cls_gt.argmax(dim=1, keepdim=True)
        cls_gt_list.append(cls_gt_result)  ## for k-fold cross validation

        cls_gt_result = cls_gt_result.squeeze().detach().cpu().numpy()    # (1, 2) --> (2)

        cls_pred_numpy = cls_pred.detach().cpu().numpy().astype('float32')
        cls_pred_prob_list.append(cls_pred_prob[0])   ## for k-fold cross validation
        
        cls_pred_prob_numpy = cls_pred_prob.detach().cpu().numpy().astype('float32')

        cls_log =  open(os.path.join(output_dir,f'cls_pred_{idx}.txt'), 'a')
        data = f'PRED:  {cls_pred_result}, TRUE:  {cls_gt_result}, LOGIT:  {cls_pred_numpy}, PROBABILITY:  {cls_pred_prob_numpy}'
        cls_log.write(data + "\n")
        cls_log.close()


        ## For Accuracy, AUROC calculation 
        correct += (cls_pred_result == cls_gt_result).sum().item()
        y_true.append(np.float32(cls_gt_result))
        y_prob.append(cls_pred_prob_numpy[:,1][0])
        y_prob_bi.append(np.float32(cls_pred_result))

        # Metric CLS
        auc_1            = auc_metric(y_pred=cls_pred_prob, y=cls_gt)

        cls_pred_prob_one_hot = torch.argmax(cls_pred_prob, dim=1)
        cls_pred_prob_one_hot = torch.nn.functional.one_hot(cls_pred_prob_one_hot, num_classes=2)
        
        confuse_matrix = confuse_metric(y_pred=cls_pred_prob_one_hot, y=cls_gt)
        
 
    ## For Accuracy, AUROC calculation 
    correct /= size

    tp, tn, fp, fn = 0, 0, 0, 0
    sensitivity, specificity, precision, f1_score = 0, 0, 0, 0
    for i in range(size):
        if y_true[i] == 0 and y_prob_bi[i] == 0:    
            tn +=1
        elif y_true[i] == 0 and y_prob_bi[i] == 1:
            fp += 1
        elif y_true[i] == 1 and y_prob_bi[i] == 0:
            fn += 1
        elif y_true[i] == 1 and y_prob_bi[i] == 1:
            tp += 1
        else:
            raise
    
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)

    ## precision denominator can equal 0 if all predicted values are negative!
    if tp+fp == 0:
        precision = -1
    else: 
        precision = tp/(tp+fp)
    
    f1_score = 2*tp/(2*tp+fp+fn)


    # Aggregatation
    auc_1                = auc_metric.aggregate()

    fpr, tpr, th = roc_curve(y_true, y_prob)
    auc_2 = auc(fpr, tpr)

    metric_logger.update(auc=auc_1)
    metric_logger.update(auc_2=auc_2)          
    metric_logger.update(acc_2=correct, sen_2=sensitivity, spe_2=specificity, pre_2=precision, f1_score_2=f1_score)
        
    auc_metric.reset()
    confuse_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}, cls_pred_prob_list, cls_gt_list
##################################################################################################


@torch.no_grad()
def infer_Up_SMART_Net_Single_CLS(model, data_loader, device, print_freq, save_dir):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)
    header = 'TEST:'
    
    save_dict = dict()
    img_path_list = []
    img_list = []
    cls_list = []
    seg_list = []
    rec_list = []
    act_list = []

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)      # (B, C, H, W, D)
        file_name = os.path.basename(batch_data['image_meta_dict']['filename_or_obj'][0])[:-7]

        # register forward hook
        model.module.encoders[-2].basic_module.non_linearity.register_forward_hook(get_activation('Activation Map')) # for Activation Map

        cls_pred = model(inputs)

        # post-processing
        cls_pred = torch.sigmoid(cls_pred)

        current_act_map = Activation_Map(activation['Activation Map'])

        img_path_list.append(batch_data['image_meta_dict']['filename_or_obj'][0])
        img_list.append(inputs.detach().cpu().squeeze())
        cls_list.append(cls_pred.detach().cpu().squeeze())
        act_list.append(current_act_map)

        sitk_img = sitk.GetImageFromArray(current_act_map)
        sitk.WriteImage(sitk_img, os.path.join(save_dir, f'{file_name}_activation_map.nii.gz'))


    save_dict['img_path_list']  = img_path_list
    save_dict['img_list']       = img_list
    save_dict['cls_pred']       = cls_list
    save_dict['activation_map'] = act_list
    np.savez(os.path.join(save_dir, f'{file_name}_result.npz'), result=save_dict) 
