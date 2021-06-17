import torch
from torch import nn
import torch.nn.functional as F

from .featpool import build_featpool
from .feat2d import build_feat2d

from .query import build_query
from .util import grad_mul_const

import math

from .loss import build_bceloss

class TCN(nn.Module):
    def __init__(self, hidden_size, ln, ks, ps):
        super(TCN, self).__init__()

        self.fc = nn.Linear(hidden_size, hidden_size)

        self.conv = nn.Conv2d(hidden_size, hidden_size, 1, padding=0)
        self.conv_res = nn.Conv2d(hidden_size, hidden_size, 1, padding=0)

        self.conv2d_1 = nn.Conv2d(hidden_size, hidden_size, kernel_size=ks, padding=ps)
        self.conv2d_2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=ks, padding=ps)
        self.conv2d_3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=ks, padding=ps)
        
        self.pred_weight = nn.Parameter(torch.Tensor(1, hidden_size).cuda(), requires_grad=True)
        self.reset_parameters(self.pred_weight)
      

    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)

    def forward(self, qry, map2d, mask, pos_emb=None):
        qry = self.fc(qry)[:,:,None,None]
        map2d = self.conv(map2d) * mask
  
        map2d_f = F.normalize(qry * map2d) * mask

        map2d = self.conv2d_1(map2d_f).relu() * mask
        map2d = self.conv2d_2(map2d).relu() * mask
        map2d = self.conv2d_3(map2d).relu() * mask
        map2d = map2d  + self.conv_res(map2d_f).relu() * mask

        scores2d = torch.mul(map2d, self.pred_weight[:,:,None,None]).sum(dim=1) 

        return scores2d            


class TCN_BASE(nn.Module):
    def __init__(self, cfg):
        super(TCN_BASE, self).__init__()

        hidden_size = cfg.MODEL.TCN.PREDICTOR.HIDDEN_SIZE
        self.hidden_size = hidden_size 

        self.feat2d = build_feat2d(cfg.MODEL.TCN.FEAT2D.POOLING_COUNTS, cfg.MODEL.TCN.NUM_CLIPS)
        self.mask = self.feat2d.mask2d

        self.NUM_CLIPS = cfg.MODEL.TCN.NUM_CLIPS

        Datasets = cfg.DATASETS.TRAIN
        self.datasets = Datasets[0].split('_')[0]

        self.featpool = build_featpool(cfg) 
   
        self.query = build_query(cfg)
        self.bceloss = build_bceloss(cfg, self.feat2d.mask2d)

        
        print("training on the dataset:  {}".format(self.datasets))
         
        self.TCN = TCN(self.hidden_size, 3, 5, 2)
   
        print('use TCN-BASE')   

    def conf_mask(self, pooling_counts, N):
        mask2d = torch.zeros(N, N, dtype=torch.bool)
        mask2d[range(N), range(N)] = 1

        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            for _ in range(c): 
                # fill a diagonal line 
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2
        

        mask2d = mask2d.to("cuda")
        return mask2d, maskij        

    def forward(self, batches, ious2d=None, flag=None):
        """
        Arguments:

        Returns:

        """
        self.mask = self.feat2d.mask2d
        query, _ = self.query(batches.queries, batches.wordlens)
        feats_set = self.featpool(batches.feats)
        feats = feats_set[0]        

        if self.datasets == 'didemo' and batches.feats.shape[-2] > self.NUM_CLIPS:
            extend_N = batches.feats.shape[-2]
            extend_poolCount = [extend_N-1]
            mask2d_extend, maskij = self.conf_mask(extend_poolCount, extend_N)
            self.mask = mask2d_extend

            map2d_o = self.feat2d(feats, maskij) * self.mask

        else:
            map2d_o = self.feat2d(feats) * self.mask    

        bz, dim, N, N = map2d_o.size()
        
        scores2d = self.TCN(query, map2d_o, self.mask)
 
        if self.training:
            loss = self.bceloss(scores2d, ious2d)
            return loss 
        else:
            return scores2d.sigmoid() * self.mask
