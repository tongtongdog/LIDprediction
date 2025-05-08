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
        super().__init__(pool, flatten, dropout, linear)                       



class ClassificationHeadClinical(nn.Sequential):

    def __init__(self, in_channels, out_channels, hidden_channels1, hidden_channels2, pooling="avg", dropout=0.2):
        
        # dropout1 = nn.Dropout(p=dropout, inplace=False) if dropout else nn.Identity()
        # linear1  = nn.Linear(in_channels, hidden_channels1, bias=True)
        # relu1    = nn.ReLU()

        # dropout2 = nn.Dropout(p=dropout, inplace=False) if dropout else nn.Identity()
        linear2  = nn.Linear(hidden_channels1, hidden_channels2, bias=True)
        relu2    = nn.ReLU()
   
        dropout3 = nn.Dropout(p=dropout, inplace=False) if dropout else nn.Identity()
        linear3  = nn.Linear(hidden_channels2, out_channels, bias=True)
        
        super().__init__(linear2, relu2, dropout3, linear3)
