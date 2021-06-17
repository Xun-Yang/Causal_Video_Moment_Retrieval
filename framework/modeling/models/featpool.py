import torch
from torch import nn
from torch.functional import F

class FeatAvgPool(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride, Datasets=None):
        super(FeatAvgPool, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.dataset = Datasets

        if self.dataset not in {'didemo'}:
            #print("not using didemo")
            self.pool = nn.AvgPool1d(kernel_size, stride)

    def forward(self, x_):       
        x = self.conv(x_.transpose(1, 2)).relu()
        
        if self.dataset not in {'didemo'}:
            return [self.pool(x)]
        else:
            return [x]            

def build_featpool(cfg):
    input_size = cfg.MODEL.TCN.FEATPOOL.INPUT_SIZE
    hidden_size = cfg.MODEL.TCN.FEATPOOL.HIDDEN_SIZE
    kernel_size = cfg.MODEL.TCN.FEATPOOL.KERNEL_SIZE
    stride = cfg.INPUT.NUM_PRE_CLIPS // cfg.MODEL.TCN.NUM_CLIPS
    Datasets = cfg.DATASETS.TRAIN
    Datasets = Datasets[0].split('_')[0]

    return FeatAvgPool(input_size, hidden_size, kernel_size, stride, Datasets)
