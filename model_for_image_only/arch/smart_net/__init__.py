import torch
from torch import nn as nn
from torch.nn import functional as F

from .seg_decoder import seg_Decoder
from ..base import modules as md


def create_seg_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, is3d):
    # create decoder path consisting of the Decoder modules. The length of the decoder list is equal to `len(f_maps) - 1`
    decoders = []
    reversed_f_maps = list(reversed(f_maps))         ## (64, 128, 256, 512, 1024) -> (1024, 512, 256, 128, 64)
    
    ## here, i is equal to [0, 1, 2, 3]
    for i in range(len(reversed_f_maps) - 1):
        if basic_module == md.DoubleConv:
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
        else:
            in_feature_num = reversed_f_maps[i]

        out_feature_num = reversed_f_maps[i + 1]

        decoder = seg_Decoder(in_feature_num, out_feature_num,
                          basic_module=basic_module,
                          conv_layer_order=layer_order,
                          conv_kernel_size=conv_kernel_size,
                          num_groups=num_groups,
                          padding=conv_padding,
                          is3d=is3d)
        decoders.append(decoder)
    return nn.ModuleList(decoders)

