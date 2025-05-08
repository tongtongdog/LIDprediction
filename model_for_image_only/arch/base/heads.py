import torch.nn as nn
from .modules import Flatten


class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, pooling="avg", dropout=0.2):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool3d(1) if pooling == 'avg' else nn.AdaptiveMaxPool3d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear  = nn.Linear(in_channels, out_channels, bias=True)
        super().__init__(pool, flatten, dropout, linear)                       ## 실행 시 이 순서로 실행되도록 setting

