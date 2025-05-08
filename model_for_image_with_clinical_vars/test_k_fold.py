import os
from pathlib import Path
import argparse
import datetime
import numpy as np
import time
import torch
import json
import random
# import functools

import utils
from create_model import create_model
from create_datasets.prepare_datasets import build_test_dataset
from engine import *
from losses import Uptask_Loss


def get_args_parser():
    parser = argparse.ArgumentParser('SMART-Net Framework Train and Test script', add_help=False)

    # Dataset parameters
    parser.add_argument('--data-folder-dir', default="/workspace", type=str, help='dataset folder dirname')    
    parser.add_argument('--excel-folder-dir', default="/workspace_clin", type=str, help='clinical variables dirname')
    parser.add_argument('--test-dataset-name', default="Custom", type=str, help='test dataset name')    
    
    # Model parameters
    parser.add_argument('--model-name', default='SMART_Net', type=str, help='model name')

    # DataLoader setting
    parser.add_argument('--num-workers', default=10, type=int)
    parser.add_argument('--pin-mem',    action='store_true', default=False, help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    
    # Setting Upstream, Downstream task
    parser.add_argument('--training-stream', default='Upstream', type=str, help='training stream')  

    # DataParrel or Single GPU train
    parser.add_argument('--multi-gpu-mode',       default='DataParallel', choices=['DataParallel', 'Single'], type=str, help='multi-gpu-mode')          
    parser.add_argument('--device',               default='cuda', help='device to use for training / testing')
    parser.add_argument('--cuda-device-order',    default='PCI_BUS_ID', type=str, help='cuda_device_order')
    parser.add_argument('--cuda-visible-devices', default='0', type=str, help='cuda_visible_devices')

    # Load checkpoint
    parser.add_argument('--from-pretrained-0',  default='',  help='pre-trained from checkpoint')
    parser.add_argument('--from-pretrained-1',  default='',  help='pre-trained from checkpoint')
    parser.add_argument('--from-pretrained-2',  default='',  help='pre-trained from checkpoint')
    parser.add_argument('--from-pretrained-3',  default='',  help='pre-trained from checkpoint')
    parser.add_argument('--from-pretrained-4',  default='',  help='pre-trained from checkpoint')
    parser.add_argument('--checkpoint-total-num', default=1, help='number of checkpoints to load for averaging')

    # Validation setting
    parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')

    # Prediction and Save setting
    parser.add_argument('--output-dir', default='', help='path where to save, empty for no saving')

    return parser


# Fix random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

def main(args):
           
    device = torch.device(args.device)

    print("Loading dataset ....")
    dataset_test = build_test_dataset(args=args)
    
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=args.pin_mem, drop_last=False)


    # Select Loss
    if args.training_stream == 'Upstream':
        criterion = Uptask_Loss(name=args.model_name)


    # Select Model
    print(f"Creating model  : {args.model_name}")
    print(f"Pretrained model: {args.from_pretrained_0}\n{args.from_pretrained_1}\n{args.from_pretrained_2}\n{args.from_pretrained_3}\n{args.from_pretrained_4}")

    start_time = time.time()

    total_probs = []
    checkpoints = [args.from_pretrained_0, args.from_pretrained_1, args.from_pretrained_2, args.from_pretrained_3, args.from_pretrained_4]

    for idx in range(int(args.checkpoint_total_num)):

        model = create_model(stream=args.training_stream, name=args.model_name)

        print(f"Loading checkpoint: {checkpoints[idx]}")
        checkpoint = torch.load(checkpoints[idx], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])        

        # Multi GPU
        if args.multi_gpu_mode == 'DataParallel':
            model = torch.nn.DataParallel(model)
            model.to(device)
        elif args.multi_gpu_mode == 'Single':
            model.to(device)
        else :
            raise Exception('Error...! args.multi_gpu_mode')  

        # TEST
        if args.training_stream == 'Upstream':

            if args.model_name == 'Up_SMART_Net':
                test_stats, cls_pred_prob_list, cls_gt_list = test_Up_SMART_Net(model, criterion, data_loader_test, device, args.print_freq, 1, args.output_dir, idx)
                total_probs.append(cls_pred_prob_list)
            elif args.model_name == 'Up_SMART_Net_Dual_CLS_REC':
                test_stats, cls_pred_prob_list, cls_gt_list = test_Up_SMART_Net_Dual_CLS_REC(model, criterion, data_loader_test, device, args.print_freq, 1, args.output_dir, idx)
                total_probs.append(cls_pred_prob_list)

            # ## Single
            elif args.model_name == 'Up_SMART_Net_Single_CLS':
                test_stats, cls_pred_prob_list, cls_gt_list = test_Up_SMART_Net_Single_CLS(model, criterion, data_loader_test, device, args.print_freq, 1, args.output_dir, idx)
                total_probs.append(cls_pred_prob_list)

        else :
            raise KeyError("Wrong training stream `{}`".format(args.training_stream))        

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
        
        if args.output_dir:
            with open(args.output_dir + f"/test_log_{idx}.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        print("Averaged test_stats: ", test_stats)

    
    # Average the logits

    stacked_total_probs = [torch.stack(x) for x in total_probs]
    avg_probs = torch.mean(torch.stack(stacked_total_probs), dim=0)

    final_pred = avg_probs.argmax(dim=1, keepdim=True)
    
    avg_probs_numpy = avg_probs.detach().cpu().numpy().astype('float32')
    cls_pred_result = final_pred.squeeze().detach().cpu().numpy()
 
    cls_gt_result = [x.squeeze().detach().cpu().numpy() for x in cls_gt_list]

    cls_log =  open(os.path.join(args.output_dir,f'cls_pred_average.txt'), 'a')
    for x in range(len(cls_gt_result)):
        data = f'PRED:  {cls_pred_result[x]}, TRUE:  {cls_gt_result[x]}, PROBABILITY:  {avg_probs_numpy[x]}'
        cls_log.write(data + "\n")
    cls_log.close()


    # Finish
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('TEST time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('SMART-Net Framework training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    os.environ["CUDA_DEVICE_ORDER"]     =  args.cuda_device_order
    os.environ["CUDA_VISIBLE_DEVICES"]  =  args.cuda_visible_devices        
    
    main(args)
