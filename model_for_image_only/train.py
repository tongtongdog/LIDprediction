import os
import argparse
import datetime
import numpy as np
import time
import torch
import json
import random
from pathlib import Path

import utils
from create_model import create_model
from create_datasets.prepare_datasets import build_dataset
from engine import *
from losses import Uptask_Loss
from optimizers import create_optim
from lr_schedulers import create_scheduler
from torch.utils.data import WeightedRandomSampler


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('SMART-Net Framework Train and Test script', add_help=False)

    # Dataset parameters
    parser.add_argument('--data-folder-dir', default="/workspace", type=str, help='dataset folder dirname')    

    # Model parameters
    parser.add_argument('--model-name', default='SMART_Net', type=str, help='model name')

    # DataLoader setting
    parser.add_argument('--batch-size',  default=20, type=int)
    parser.add_argument('--num-workers', default=10, type=int)
    parser.add_argument('--pin-mem',    action='store_true', default=False, help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # Optimizer parameters
    parser.add_argument('--optimizer', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    
    # Learning rate and schedule and Epoch parameters
    parser.add_argument('--lr-scheduler', default='poly_lr', type=str, metavar='lr_scheduler', help='lr_scheduler (default: "poly_learning_rate"')
    parser.add_argument('--epochs', default=1000, type=int, help='Upstream 1000 epochs, Downstream 500 epochs')  
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument("--optim-lr", default=1e-4, type=float)
    parser.add_argument('--early-stop-epoch', type=int, default=300, metavar='N', help='early stop epoch')


    # Setting Upstream, Downstream task
    parser.add_argument('--training-stream', default='Upstream', type=str, help='training stream')  

    # DataParrel or Single GPU train
    parser.add_argument('--multi-gpu-mode',       default='DataParallel', choices=['DataParallel', 'Single'], type=str, help='multi-gpu-mode')          
    parser.add_argument('--device',               default='cuda', help='device to use for training / testing')
    parser.add_argument('--cuda-device-order',    default='PCI_BUS_ID', type=str, help='cuda_device_order')
    parser.add_argument('--cuda-visible-devices', default='0', type=str, help='cuda_visible_devices')

    # Continue Training
    parser.add_argument('--resume',           default='',  help='resume from checkpoint')  # '' = None
    parser.add_argument('--from-pretrained-0',  default='',  help='pre-trained from checkpoint')
    parser.add_argument('--load-weight-type', default='full',  help='the types of loading the pre-trained weights')
    parser.add_argument('--freeze-feature-extractor', type= str2bool, default=False, help='whether to freeze loaded weights')

    # Validation setting
    parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')

    # Prediction and Save setting
    parser.add_argument('--output-dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--output-dir-2', default='', help='path where to save pred results')

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
           
    utils.print_args(args)
    device = torch.device(args.device)

    ## Dataset and Dataloader
    print("Loading dataset ....")

    dataset_train, weight_vector = build_dataset(is_train=True,  args=args)   
    dataset_valid, _ = build_dataset(is_train=False, args=args)

    ## weighted loader ##
    # sampler = WeightedRandomSampler(weight_vector, len(weight_vector), replacement=True)
    # data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=args.pin_mem, drop_last=True)

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,  pin_memory=args.pin_mem, drop_last=False)
    data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1,               num_workers=args.num_workers, shuffle=False, pin_memory=args.pin_mem, drop_last=False)



    # Select Model
    print(f"Creating model  : {args.model_name}")
    model = create_model(stream=args.training_stream, name=args.model_name)
    print(model)
    

    # Optimizer & LR Scheduler
    optimizer    = create_optim(name=args.optimizer, model=model, args=args)
    lr_scheduler = create_scheduler(name=args.lr_scheduler, optimizer=optimizer, args=args)


    # Resume
    if args.resume:
        print("Loading... Resume")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])        
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])        
        args.start_epoch = checkpoint['epoch'] + 1  
        try:
            log_path = os.path.dirname(args.resume)+'/log.txt'
            lines    = open(log_path,'r').readlines()
            val_loss_list = []
            for l in lines:
                exec('log_dict='+l.replace('NaN', '0'))
                val_loss_list.append(log_dict['valid_loss'])
            print("Epoch: ", np.argmin(val_loss_list), " Minimum Val Loss ==> ", np.min(val_loss_list))
        except:
            pass

        # Optimizer Error fix...!
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()



    # Using the pre-trained feature extract's weights
    if args.from_pretrained_0:
       
        ## loading pre-trained weights from tasks done before
        print("Loading... Pre-trained")      
        model_dict = model.state_dict() 

        attr_name = f'from_pretrained_0'
        attr_value = getattr(args, attr_name, None)
        checkpoint = torch.load(attr_value, map_location='cpu')
        model_state_dict = checkpoint['model_state_dict']

        if args.load_weight_type == 'full':
            model.load_state_dict(model_state_dict)
        
        elif args.load_weight_type == 'encoder':

            filtered_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if (k in model_dict) and ('encoders.' in k or 'rec_' in k)}
            
            model_dict.update(filtered_dict)             
            model.load_state_dict(model_dict, strict=False)   



    # Multi GPU
    if args.multi_gpu_mode == 'DataParallel':
        model = torch.nn.DataParallel(model)
        model.to(device)
    elif args.multi_gpu_mode == 'Single':
        model.to(device)
    else :
        raise Exception('Error...! args.multi_gpu_mode')    


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()


    os.makedirs(args.output_dir_2)

    cls_log =  open(os.path.join(args.output_dir_2, 'cls_pred.txt'), 'w')
    cls_log.close()


    # Whole LOOP
    for epoch in range(args.start_epoch, args.epochs):

        # early stop
        if epoch == args.early_stop_epoch:
            break

        # Select Loss
        criterion = Uptask_Loss(name=args.model_name, epoch=epoch)

        # Train & Valid
        if args.training_stream == 'Upstream':

            if args.model_name == 'Up_SMART_Net':

                train_stats = train_Up_SMART_Net(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size)
                print("Averaged train_stats: ", train_stats)
                
                valid_stats = valid_Up_SMART_Net(model, criterion, data_loader_valid, device, epoch, args.print_freq, 1, args.output_dir_2)
                print("Averaged valid_stats: ", valid_stats)
            
            ## Dual    
            elif args.model_name == 'Up_SMART_Net_Dual_CLS_SEG':
                train_stats = train_Up_SMART_Net_Dual_CLS_SEG(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size)
                print("Averaged train_stats: ", train_stats)

                valid_stats = valid_Up_SMART_Net_Dual_CLS_SEG(model, criterion, data_loader_valid, device, epoch, args.print_freq, 1, args.output_dir_2)
                print("Averaged valid_stats: ", valid_stats)

            elif args.model_name == 'Up_SMART_Net_Dual_CLS_REC':
                train_stats = train_Up_SMART_Net_Dual_CLS_REC(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size, args.freeze_feature_extractor)
                print("Averaged train_stats: ", train_stats)
                
                valid_stats = valid_Up_SMART_Net_Dual_CLS_REC(model, criterion, data_loader_valid, device, epoch, args.print_freq, 1, args.output_dir_2)
                print("Averaged valid_stats: ", valid_stats)
            
            elif args.model_name == 'Up_SMART_Net_Dual_SEG_REC':
                train_stats = train_Up_SMART_Net_Dual_SEG_REC(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size)
                print("Averaged train_stats: ", train_stats)

                valid_stats = valid_Up_SMART_Net_Dual_SEG_REC(model, criterion, data_loader_valid, device, epoch, args.print_freq, 1, args.output_dir_2)
                print("Averaged valid_stats: ", valid_stats)


            ## Single
            elif args.model_name == 'Up_SMART_Net_Single_CLS':
                train_stats = train_Up_SMART_Net_Single_CLS(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size)
                print("Averaged train_stats: ", train_stats)
                
                valid_stats = valid_Up_SMART_Net_Single_CLS(model, criterion, data_loader_valid, device, epoch, args.print_freq, 1, args.output_dir_2)
                print("Averaged valid_stats: ", valid_stats)

            else : 
                raise KeyError("Wrong model name `{}`".format(args.model_name))     


        else :
            raise KeyError("Wrong training stream `{}`".format(args.training_stream))        



        # Save & Prediction png
        checkpoint_paths = args.output_dir + '/epoch_' + str(epoch) + '_checkpoint.pth'
        torch.save({
            'model_state_dict': model.state_dict() if args.multi_gpu_mode == 'Single' else model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }, checkpoint_paths)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'valid_{k}': v for k, v in valid_stats.items()},
                    'epoch': epoch}
        
        if args.output_dir:
            with open(args.output_dir + "/log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        lr_scheduler.step(epoch)


    # Finish
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('SMART-Net Framework training and evaluation script', parents=[get_args_parser()])
    args   = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    os.environ["CUDA_DEVICE_ORDER"]     =  args.cuda_device_order
    os.environ["CUDA_VISIBLE_DEVICES"]  =  args.cuda_visible_devices        
    
    main(args)
