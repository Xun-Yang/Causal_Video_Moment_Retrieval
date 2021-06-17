import torch
from torch import nn

class SparseMaxPool(nn.Module):
    def __init__(self, pooling_counts, N):
        super(SparseMaxPool, self).__init__()
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
        
        
        self.poolers = nn.ModuleList()
        poolers = [nn.MaxPool1d(2,1) for _ in range(pooling_counts[0])]
        #poolers = [nn.AvgPool1d(2,1) for _ in range(pooling_counts[0])]
        #[nn.Conv1d(512, 512, 2,1) for _ in range(pooling_counts[0])]
        self.poolers.extend(poolers)
        for c in pooling_counts[1:]:
            rest_lay = [nn.MaxPool1d(3,2)] + [nn.MaxPool1d(2,1) for _ in range(c - 1)]
            #rest_lay = [nn.AvgPool1d(3,2)] + [nn.AvgPool1d(2,1) for _ in range(c - 1)]
            #[nn.Conv1d(512, 512, 3, 2)] + [nn.Conv1d(512, 512, 2, 1) for _ in range(c - 1)]
            self.poolers.extend(rest_lay)

        self.mask2d = mask2d.to("cuda")
        self.maskij = maskij

        

    def forward(self, x, maskij_=None):
        B, D, N = x.shape
        map2d = x.new_zeros(B, D, N, N)
        map2d[:, :, range(N), range(N)] = x

        if maskij_ is None:
            maskij_ = self.maskij
         
        for pooler, (i, j) in zip(self.poolers, maskij_):
            
            x = pooler(x)
            map2d[:, :, i, j] = x
        return map2d

def build_feat2d(pooling_counts, num_clips):

    return SparseMaxPool(pooling_counts, num_clips)
