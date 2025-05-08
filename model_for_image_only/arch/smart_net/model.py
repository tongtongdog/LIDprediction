from typing import Optional, Union, List


from .rec_decoder import AE_Decoder
from ..encoders import create_encoders             
from . import create_seg_decoders

from ..base import modules as md

from ..base import ClassificationHead

import torch 
import torch.nn as nn

import time
import os
import numpy as np

## Smart-Net
class Up_SMART_Net(torch.nn.Module):
    def __init__(
        self,

        # the original version of the model
        # encoder_channels: tuple = (64, 128, 256, 512, 1024),
        # seg_decoder_channels: tuple = (64, 128, 256, 512, 1024),    ## because of the way create_seg_decoder works, the channel size is reversed
        # rec_decoder_channels: tuple = (512, 256, 128, 64),

        # reduding the depth of the model
        encoder_channels: tuple = (64, 128, 256, 512),
        seg_decoder_channels: tuple = (64, 128, 256, 512),
        rec_decoder_channels: tuple = (256, 128, 64),

        decoder_attention_type: Optional[str] = None,
        in_channels: int = 1
    ):
        super().__init__()
        
        self.encoders = create_encoders(
            in_channels=in_channels,              ## in_channels (int): number of input channels
            f_maps = encoder_channels,            ## Depth of the encoder is equal to `len(f_maps)`     
            # basic_module = md.ResNetBlockSE,
            basic_module = md.ResNetBlock,
            # basic_module = md.DoubleConv,         
            conv_kernel_size = 3,                 ## conv_kernel_size (int or tuple): size of the convolving kernel
            conv_padding = 1,                     ## padding (int or tuple): add zero-padding added to all three sides of the input
            layer_order = 'cbl',                  ## conv_layer_order (string): determines the order of layers in `DoubleConv` module. See `DoubleConv` for more info.
            num_groups = 8,                       ## num_groups (int): number of groups for the GroupNorm, number of channels in input to be divisible by num_groups
            pool_kernel_size = 2,                 ## pool_kernel_size (int or tuple): the size of the window for max or average pooling
            is3d = True
        )



        ###### CLS ###### 
        ## result is a logit value
        self.classification_head = ClassificationHead(
            in_channels=encoder_channels[-1],
            out_channels=2,     # 1 -> 2
            pooling='avg', 
            dropout=0.5            
        )


        ###### SEG ######
        ## result is a logit value
        self.seg_decoder = create_seg_decoders(
            f_maps = seg_decoder_channels, 
            basic_module = md.DoubleConv, 
            conv_kernel_size = 3, 
            conv_padding = 1, 
            layer_order = 'bcr', 
            num_groups = 8, 
            is3d = True
        )

        self.seg_final_conv = nn.Conv3d(seg_decoder_channels[0], out_channels=1, kernel_size=1)

        ###### REC ######
        self.rec_decoder = AE_Decoder(
            encoder_channels=encoder_channels,              ## (64, 128, 256, 512, 1024)
            decoder_channels=rec_decoder_channels,          ## (512, 256, 128, 64)
            center=False,
            attention_type=decoder_attention_type,
        )

        self.rec_final_conv = nn.Conv3d(rec_decoder_channels[-1], out_channels=1, kernel_size=1)

 
        self.name = "SMART-Net-{}".format('resnet_3d')

    def forward(self, x):

        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            # inserts as the very first element each time
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list -> the last output 1024 (we don't need this for skip connections)
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        labels = self.classification_head(x)


        # Seg
        seg_input = x
        for decoder, encoder_features in zip(self.seg_decoder, encoders_features):
            # pass the output from the corresponding encoder and the output of the previous decoder
            seg_input = decoder(encoder_features, seg_input)
        masks = self.seg_final_conv(seg_input)

        # Rec
        rec_decoder_output = self.rec_decoder(x)
        restores = self.rec_final_conv(rec_decoder_output)

        return labels, masks, restores


## Dual

## CLS+SEG
class Up_SMART_Net_Dual_CLS_SEG(torch.nn.Module):
    def __init__(
        self,
        encoder_channels: tuple = (64, 128, 256, 512, 1024),
        seg_decoder_channels: tuple = (64, 128, 256, 512, 1024),    ## because of the way create_seg_decoder works, the channel size is reversed
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 1,
    ):
        super().__init__()

        self.encoders = create_encoders(
            in_channels=in_channels,              ## in_channels (int): number of input channels
            f_maps = encoder_channels,            ## Depth of the encoder is equal to `len(f_maps)`     
            basic_module = md.ResNetBlockSE,
            # basic_module = md.ResNetBlock,
            # basic_module = md.DoubleConv,      
            conv_kernel_size = 3,                 ## conv_kernel_size (int or tuple): size of the convolving kernel
            conv_padding = 1,                     ## padding (int or tuple): add zero-padding added to all three sides of the input
            layer_order = 'cbr',                  ## conv_layer_order (string): determines the order of layers in `DoubleConv` module. See `DoubleConv` for more info.
            num_groups = 8,                       ## num_groups (int): number of groups for the GroupNorm, number of channels in input to be divisible by num_groups
            pool_kernel_size = 2,                 ## pool_kernel_size (int or tuple): the size of the window for max or average pooling
            is3d = True
        )

        ###### CLS ###### 
        ## result is a logit value
        self.classification_head = ClassificationHead(
            in_channels=encoder_channels[-1],
            out_channels=2,     # 1 -> 2
            pooling='avg', 
            dropout=0.5            
        )
        
        ###### SEG ######
        ## result is a logit value
        self.seg_decoder = create_seg_decoders(
            f_maps = seg_decoder_channels, 
            basic_module = md.DoubleConv, 
            conv_kernel_size = 3, 
            conv_padding = 1, 
            layer_order = 'bcr', 
            num_groups = 8, 
            is3d = True
        )

        self.seg_final_conv = nn.Conv3d(seg_decoder_channels[0], out_channels=1, kernel_size=1)

        self.name = "SMART-Net-{}".format('resnet_3d')

    def forward(self, x):

        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            # inserts as the very first element each time
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list -> the last output 1024 (we don't need this for skip connections)
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # Cls
        labels = self.classification_head(x)

        # Seg
        seg_input = x
        for decoder, encoder_features in zip(self.seg_decoder, encoders_features):
            # pass the output from the corresponding encoder and the output of the previous decoder
            seg_input = decoder(encoder_features, seg_input)
        masks = self.seg_final_conv(seg_input)

        return labels, masks



## CLS+REC
class Up_SMART_Net_Dual_CLS_REC(torch.nn.Module):
    def __init__(
        self,
        # encoder_channels: tuple = (64, 128, 256, 512, 1024),
        # rec_decoder_channels: tuple = (512, 256, 128, 64),

        # reduding the depth of the model
        encoder_channels: tuple = (64, 128, 256, 512),
        rec_decoder_channels: tuple = (256, 128, 64),

        decoder_attention_type: Optional[str] = None,
        in_channels: int = 1,
    ):
        super().__init__()

        self.encoders = create_encoders(
            in_channels=in_channels,              ## in_channels (int): number of input channels
            f_maps = encoder_channels,            ## Depth of the encoder is equal to `len(f_maps)`     
            # basic_module = md.ResNetBlockSE,
            basic_module = md.ResNetBlock,
            # basic_module = md.DoubleConv,      
            conv_kernel_size = 3,                 ## conv_kernel_size (int or tuple): size of the convolving kernel
            conv_padding = 1,                     ## padding (int or tuple): add zero-padding added to all three sides of the input
            layer_order = 'cbl',                  ## conv_layer_order (string): determines the order of layers in `DoubleConv` module. See `DoubleConv` for more info.
            num_groups = 8,                       ## num_groups (int): number of groups for the GroupNorm, number of channels in input to be divisible by num_groups
            pool_kernel_size = 2,                 ## pool_kernel_size (int or tuple): the size of the window for max or average pooling
            is3d = True
        )

        ###### CLS ###### 
        ## result is a logit value
        self.classification_head = ClassificationHead(
            in_channels=encoder_channels[-1],
            out_channels=2,     # 1 -> 2
            pooling='avg', 
            dropout=0.5            
        )

        ###### REC ######
        self.rec_decoder = AE_Decoder(
            encoder_channels=encoder_channels,              ## (64, 128, 256, 512, 1024)
            decoder_channels=rec_decoder_channels,          ## (512, 256, 128, 64)
            center=False,
            attention_type=decoder_attention_type,
        )

        self.rec_final_conv = nn.Conv3d(rec_decoder_channels[-1], out_channels=1, kernel_size=1)

        self.name = "SMART-Net-{}".format('resnet_3d')


    def forward(self, x):

        encoders_features = []
        for encoder in self.encoders:


            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            # inserts as the very first element each time
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list -> the last output 1024 (we don't need this for skip connections)
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # Cls
        labels = self.classification_head(x)

        # Rec
        rec_decoder_output = self.rec_decoder(x)
        restores = self.rec_final_conv(rec_decoder_output)

        return labels, restores


## SEG + REC
class Up_SMART_Net_Dual_SEG_REC(torch.nn.Module):
    def __init__(
        self,
        encoder_channels: tuple = (64, 128, 256, 512, 1024),
        seg_decoder_channels: tuple = (64, 128, 256, 512, 1024),    ## because of the way create_seg_decoder works, the channel size is reversed
        rec_decoder_channels: tuple = (512, 256, 128, 64),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 1,
    ):
        super().__init__()

        self.encoders = create_encoders(
            in_channels=in_channels,              ## in_channels (int): number of input channels
            f_maps = encoder_channels,            ## Depth of the encoder is equal to `len(f_maps)`     
            basic_module = md.ResNetBlockSE,
            # basic_module = md.ResNetBlock,
            # basic_module = md.DoubleConv,        
            conv_kernel_size = 3,                 ## conv_kernel_size (int or tuple): size of the convolving kernel
            conv_padding = 1,                     ## padding (int or tuple): add zero-padding added to all three sides of the input
            layer_order = 'cbr',                  ## conv_layer_order (string): determines the order of layers in `DoubleConv` module. See `DoubleConv` for more info.
            num_groups = 8,                       ## num_groups (int): number of groups for the GroupNorm, number of channels in input to be divisible by num_groups
            pool_kernel_size = 2,                 ## pool_kernel_size (int or tuple): the size of the window for max or average pooling
            is3d = True
        )


        ###### SEG ######
        ## result is a logit value
        self.seg_decoder = create_seg_decoders(
            f_maps = seg_decoder_channels, 
            basic_module = md.DoubleConv, 
            conv_kernel_size = 3, 
            conv_padding = 1, 
            layer_order = 'bcr', 
            num_groups = 8, 
            is3d = True
        )

        self.seg_final_conv = nn.Conv3d(seg_decoder_channels[0], out_channels=1, kernel_size=1)


        ###### REC ######
        self.rec_decoder = AE_Decoder(
            encoder_channels=encoder_channels,              ## (64, 128, 256, 512, 1024)
            decoder_channels=rec_decoder_channels,          ## (512, 256, 128, 64)
            center=False,
            attention_type=decoder_attention_type,
        )

        self.rec_final_conv = nn.Conv3d(rec_decoder_channels[-1], out_channels=1, kernel_size=1)

        self.name = "SMART-Net-{}".format('resnet_3d')

    def forward(self, x):

        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            # inserts as the very first element each time
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list -> the last output 1024 (we don't need this for skip connections)
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # Seg
        seg_input = x
        for decoder, encoder_features in zip(self.seg_decoder, encoders_features):
            # pass the output from the corresponding encoder and the output of the previous decoder
            seg_input = decoder(encoder_features, seg_input)
        masks = self.seg_final_conv(seg_input)


        # Rec
        rec_decoder_output = self.rec_decoder(x)
        restores = self.rec_final_conv(rec_decoder_output)

        return masks, restores
    



## Single
    # CLS
class Up_SMART_Net_Single_CLS(torch.nn.Module):
    def __init__(
        self,
        encoder_channels: tuple = (64, 128, 256, 512),
        in_channels: int = 1,
    ):
        super().__init__()

        self.encoders = create_encoders(
            in_channels=in_channels,              ## in_channels (int): number of input channels
            f_maps = encoder_channels,            ## Depth of the encoder is equal to `len(f_maps)`     
            # basic_module = md.ResNetBlockSE,
            basic_module = md.ResNetBlock,
            # basic_module = md.DoubleConv,        
            conv_kernel_size = 3,                 ## conv_kernel_size (int or tuple): size of the convolving kernel
            conv_padding = 1,                     ## padding (int or tuple): add zero-padding added to all three sides of the input
            layer_order = 'cbl',                  ## conv_layer_order (string): determines the order of layers in `DoubleConv` module. See `DoubleConv` for more info.
            num_groups = 8,                       ## num_groups (int): number of groups for the GroupNorm, number of channels in input to be divisible by num_groups
            pool_kernel_size = 2,                 ## pool_kernel_size (int or tuple): the size of the window for max or average pooling
            is3d = True
        )

        ###### CLS ###### 
        ## result is a logit value
        self.classification_head = ClassificationHead(
            in_channels=encoder_channels[-1],
            out_channels=2,     # 1 -> 2
            pooling='avg', 
            dropout=0.5            
        )

        self.name = "SMART-Net-{}".format('resnet_3d')


    def forward(self, x):

        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            # inserts as the very first element each time
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list -> the last output 1024 (we don't need this for skip connections)
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # Cls
        labels = self.classification_head(x)

        return labels
