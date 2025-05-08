from typing import Optional, Union, List

from .rec_decoder import AE_Decoder
from ..encoders import create_encoders             
from . import create_seg_decoders

from ..base import modules as md


from ..base import ClassificationHead, ClassificationHeadClinical

import torch 
import torch.nn as nn



class Up_SMART_Net(torch.nn.Module):
    def __init__(
        self,
      
        encoder_channels: tuple = (64, 128, 256, 512),
        seg_decoder_channels: tuple = (64, 128, 256, 512),
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


        ###### CLS + Clinical ###### 
        self.classification_head = ClassificationHeadClinical(
            in_channels=encoder_channels[-1],    # depending on the length of the clinical variable vector
            out_channels=2,             # 1 -> 2
            hidden_channels1 = 32+18,      # need to change this
            hidden_channels2 = 32,      # need to change this
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
            # use_batchnorm=False,
            center=False,
            attention_type=decoder_attention_type,
        )

        self.rec_final_conv = nn.Conv3d(rec_decoder_channels[-1], out_channels=1, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = md.Flatten()
        self.concat = md.ConcatenateWithVector()
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        
        self.linear1 = nn.Linear(512, 32, bias=True)
        self.relu1    = nn.ReLU()

        self.linear2 = nn.Linear(512, 256, bias=True)
        self.relu2    = nn.ReLU()

        self.linear3 = nn.Linear(14, 32, bias=True)
        self.relu3    = nn.ReLU()


        self.name = "SMART-Net-{}".format('resnet_3d')

    def forward(self, x, additional_vector):

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
        cls_input = x
        cls_input = self.pool(cls_input)
        cls_input = self.flatten(cls_input)
        cls_input = self.relu1(self.linear1(cls_input))

        additional_vector = additional_vector.float()

        cls_input = self.concat(cls_input, additional_vector)
        labels = self.classification_head(cls_input)

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

## CLS+REC
class Up_SMART_Net_Dual_CLS_REC(torch.nn.Module):
    def __init__(
        self,
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
        self.classification_head = ClassificationHeadClinical(
            in_channels=encoder_channels[-1],    # depending on the length of the clinical variable vector
            out_channels=2,             # 1 -> 2
            hidden_channels1 = 32+18,      # need to change this
            # hidden_channels1 = 32+19,      # LEDD variable added
            hidden_channels2 = 32,      # need to change this
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

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = md.Flatten()
        self.concat = md.ConcatenateWithVector()
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        
        self.linear1 = nn.Linear(512, 32, bias=True)
        self.relu1    = nn.ReLU()

        self.linear2 = nn.Linear(512, 256, bias=True)
        self.relu2    = nn.ReLU()

        self.linear3 = nn.Linear(14, 32, bias=True)
        self.relu3    = nn.ReLU()

        self.name = "SMART-Net-{}".format('resnet_3d')


    def forward(self, x, additional_vector):

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
        cls_input = x
        cls_input = self.pool(cls_input)
        cls_input = self.flatten(cls_input)
        cls_input = self.relu1(self.linear1(cls_input))

        additional_vector = additional_vector.float()

        cls_input = self.concat(cls_input, additional_vector)
        labels = self.classification_head(cls_input)

        # Rec
        rec_decoder_output = self.rec_decoder(x)
        restores = self.rec_final_conv(rec_decoder_output)

        return labels, restores



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
        self.classification_head = ClassificationHeadClinical(
            in_channels=encoder_channels[-1],    # depending on the length of the clinical variable vector
            out_channels=2,             # 1 -> 2
            hidden_channels1 = 32+18,      # need to change this
            hidden_channels2 = 32,      # need to change this
            pooling='avg',
            dropout=0                  
        )

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = md.Flatten()
        self.concat = md.ConcatenateWithVector()
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        
        self.linear1 = nn.Linear(512, 32, bias=True)
        self.relu1    = nn.ReLU()

        self.linear2 = nn.Linear(512, 256, bias=True)
        self.relu2    = nn.ReLU()

        self.linear3 = nn.Linear(14, 32, bias=True)
        self.relu3    = nn.ReLU()

        self.name = "SMART-Net-{}".format('resnet_3d')


    def forward(self, x, additional_vector):

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
        cls_input = x
        cls_input = self.pool(cls_input)
        cls_input = self.flatten(cls_input)
        cls_input = self.relu1(self.linear1(cls_input))

        additional_vector = additional_vector.float()

        cls_input = self.concat(cls_input, additional_vector)
        labels = self.classification_head(cls_input)

        return labels

