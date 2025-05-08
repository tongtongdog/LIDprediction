import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import modules as md

class UpsampleBlock(nn.Module):
    def __init__(self, scale, input_channels, output_channels, ksize=1):
        super(UpsampleBlock, self).__init__()
        
        self.upsample = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=1, padding=ksize//2),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, input):
        return self.upsample(input)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            attention_type=None,
    ):
        super().__init__()

        self.upsample   = UpsampleBlock(scale=2, input_channels=in_channels, output_channels=in_channels)

        self.attention1 = md.Attention(attention_type, in_channels=in_channels)
        self.conv       = md.DoubleConv(in_channels=in_channels, out_channels=out_channels, encoder=False, kernel_size=3, order='bcr', num_groups=8, padding=1, is3d=True)
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)
        

    def forward(self, x):
        x = self.upsample(x)

        x = self.attention1(x)
        x = self.conv(x)
        x = self.attention2(x)
        return x




class Last_DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.SingleConv(in_channels, out_channels, kernel_size=3, order='cbr', num_groups=8, padding=1, is3d=True)                       ## gcr
        self.attention1 = md.Attention(attention_type, in_channels=in_channels)
        self.conv2 = md.SingleConv(out_channels, out_channels, kernel_size=3, order='cbr', num_groups=8, padding=1, is3d=True)                      ## gcr
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x):
        x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        conv1 = md.SingleConv(in_channels, out_channels)
        conv2 = md.SingleConv(out_channels, out_channels)
        super().__init__(conv1, conv2)



class AE_Decoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            attention_type=None,
            center=False
    ):
        super().__init__()
        encoder_channels = encoder_channels[1:]    # remove first skip with same spatial resolution --> (128, 256, 512, 1024)
        
        in_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder --> (1024, 512, 256, 128)
        head_channels = in_channels[0]        ## 1024
        out_channels  = decoder_channels    ## (512, 256, 128, 64)

        if center:
            self.center = CenterBlock(head_channels, head_channels)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        blocks = [ DecoderBlock(in_ch, out_ch, attention_type) for in_ch, out_ch in zip(in_channels, out_channels) ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features):
       
        x = self.center(features)
        
        for decoder_block in self.blocks:
            x = decoder_block(x)

        return x
    