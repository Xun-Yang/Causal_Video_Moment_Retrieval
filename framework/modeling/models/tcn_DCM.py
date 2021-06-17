import torch
from torch import nn
import torch.nn.functional as F

from .featpool import build_featpool
from .feat2d import build_feat2d

from .query import build_query
from .util import dcor
from .position_encoding import build_position_encoding
import math

from .loss import build_bceloss

class BD_Adjust_Att(nn.Module):
    def __init__(self, in_size, hidden, head_num, mask, eta=1.0):
        super(BD_Adjust_Att, self).__init__()
        self.input_size = in_size
        self.hidden_size = hidden
        self.head_num = head_num
        self.head_dim = int(self.hidden_size/self.head_num)

        self.fc_value = nn.Linear(self.input_size,  self.hidden_size, bias=True)
        self.fc_key   = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_qry   = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.dp = nn.Dropout(p=0.1)
        self.eta_ = eta
        self.pred_sigma = 1.0 #nn.Parameter(torch.Tensor([1.0]).cuda(), requires_grad=True)
        

    def forward(self, qry_in, pos_input, mask):
        bz, dim, N, N = pos_input.size()
        mask = mask.view(-1)

        pos_emd =   pos_input.view(bz, dim, -1)[:,:,mask]
        output = pos_input.new_zeros(bz, dim, N*N)
        

        if len(qry_in.shape) ==2:
            qry_in = qry_in[:,:,None]
        else:
            qry_in =     qry_in.view(bz, dim, -1)[:,:,mask]

        value =   self.fc_value(pos_emd.transpose(1,2)).relu()
        key =     self.fc_key(pos_emd.transpose(1,2)) 
        qry =     self.fc_qry(qry_in.transpose(1,2))

        qry_ = qry.view(bz, -1, self.head_num, self.head_dim).transpose(1, 2) 
        key_ = key.view(bz, -1, self.head_num, self.head_dim).permute(0, 2, 3, 1) 
        val_ = value.view(bz, -1, self.head_num, self.head_dim).transpose(1, 2)
  
        DProudct = torch.matmul(qry_, key_) / (self.head_dim ** 0.5) 
        DProudct = self.dp(F.softmax(DProudct, -1))#

        val_attent = torch.matmul(DProudct, val_).transpose(-1,-2).contiguous().view(bz, dim, DProudct.shape[-2])

        if val_attent.shape[-1] == 1:
            val_attent = val_attent.expand(-1, -1, value.shape[-2])

        output[:,:,mask] = val_attent + self.eta_ * self.pred_sigma * value.transpose(1, 2)
        output = output.view(bz, dim, N, N)
        return output.relu()

class TCN(nn.Module):
    def __init__(self, hidden_size, ln, ks, ps, mask=None, eta=1.0, do=1.0):
        super(TCN, self).__init__()

        self.fc = nn.Linear(hidden_size, hidden_size)
        self.conv     = nn.Conv2d(hidden_size, hidden_size, 1, padding=0)
        self.conv_pos = nn.Conv2d(hidden_size, hidden_size, 1, padding=0)
        self.conv_res = nn.Conv2d(hidden_size, hidden_size, 1, padding=0)  

        self.conv2d_1 = nn.Conv2d(hidden_size, hidden_size, kernel_size=ks, stride=1, padding=2, dilation=1)
        self.conv2d_2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=ks, stride=1, padding=2, dilation=1)
        self.conv2d_3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=ks, stride=1, padding=2, dilation=1)

        self.pred_weight = nn.Parameter(torch.Tensor(1, hidden_size).cuda(), requires_grad=True)
        self.reset_parameters(self.pred_weight)

        self.do = do
        if self.do > 0.0:
            self.BD_Adjust_Att = BD_Adjust_Att(hidden_size, hidden_size, 4, mask, eta)# 4 for charades and dedimo
            self.Q = nn.Linear(hidden_size, hidden_size)
            self.M = nn.Conv2d(hidden_size, hidden_size, 1, padding=0)      

    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)

    def forward(self, qry, map2d, mask, map2d_pos=None):
        
        bz, dim, N, N = map2d.size()
        map2d_feat = self.conv(map2d)* mask
        map2d_pos = self.conv_pos(map2d).tanh() * mask

        if self.do > 0.0:
            query_ = self.Q(qry.relu())[:,:,None,None] + self.M(map2d_feat.relu())
            output = self.BD_Adjust_Att(query_ * mask, map2d_pos, mask) #
            map2d_comb = map2d_feat + output * mask
        else:
            map2d_comb = map2d_feat + map2d_pos

        qry = self.fc(qry)
        qry = qry[:,:,None,None]
        map2d_f = F.normalize(qry * map2d_comb) * mask
        
        map2d = F.relu(self.conv2d_1(map2d_f)) * mask
        map2d = F.relu(self.conv2d_2(map2d)) * mask
        map2d = F.relu(self.conv2d_3(map2d)) * mask

        map2d = map2d + F.relu(self.conv_res(map2d_f)) * mask

        scores2d = torch.mul(map2d, self.pred_weight[:,:,None,None]).sum(dim=1) 

        return scores2d, map2d_pos, map2d_feat          


class TCN_DCM(nn.Module):
    def __init__(self, cfg):
        super(TCN_DCM, self).__init__()

        hidden_size = cfg.MODEL.TCN.PREDICTOR.HIDDEN_SIZE
        self.lambda_dcr = cfg.MODEL.LAMBDA
        self.gamma = cfg.MODEL.GAMMA
        self.sigma = cfg.MODEL.SIGMA
        self.eta = cfg.MODEL.ETA
        self.do = cfg.MODEL.DO

        self.hidden_size = hidden_size 
        self.feat2d = build_feat2d(cfg.MODEL.TCN.FEAT2D.POOLING_COUNTS, cfg.MODEL.TCN.NUM_CLIPS)
        self.mask = self.feat2d.mask2d

        self.NUM_CLIPS = cfg.MODEL.TCN.NUM_CLIPS

        Datasets = cfg.DATASETS.TRAIN

        self.datasets = Datasets[0].split('_')[0]
        self.featpool = build_featpool(cfg) 
        self.query = build_query(cfg)
        self.bceloss = build_bceloss(cfg, self.feat2d.mask2d)

        self.position_embedding = build_position_encoding('sine', self.hidden_size, cfg.MODEL.TCN.NUM_CLIPS)

        print("training on the dataset:  {}".format(self.datasets))
        self.TCN = TCN(self.hidden_size, 3, 5, 2, self.feat2d.mask2d, self.eta, self.do)
   
        print('use TCN_DCM')
        self.pdist = nn.PairwiseDistance(p=2, keepdim=False)  
        

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
        mask = self.mask.new_ones(self.mask.shape)
        pos_emd = self.position_embedding(mask[None, :, :]).to("cuda")
        pos_emd = pos_emd.expand(bz, dim, -1, -1) * self.mask
       
        
        if self.training:
            scores2d, mom_pos_feat, mom_vid = self.TCN(query, map2d_o, self.mask)
            loss = self.bceloss(scores2d, ious2d) # Main Loss
            feats_neg = self.featpool(batches.feats_neg)[0]

            if self.datasets == 'didemo' and batches.feats.shape[-2] > self.NUM_CLIPS:
                map2d_neg = self.feat2d(feats_neg, maskij) * self.mask
            else:    
                map2d_neg = self.feat2d(feats_neg) * self.mask

            if self.gamma > 0.0:
                scores2d_neg, _, _ = self.TCN(query, map2d_neg, self.mask)
                ious_neg = ious2d.new_zeros(bz, N, N)   

                loss_neg = self.bceloss(scores2d_neg, ious_neg) # negative query-video pairs
                loss = torch.add(loss, self.gamma * loss_neg)    
            

            vid_con = mom_vid.reshape(bz, dim, -1)[:,:,self.mask.view(-1)].transpose(1, 2).reshape(-1, dim) 
            vid_pos = mom_pos_feat.reshape(bz, dim, -1)[:,:,self.mask.view(-1)].transpose(1, 2).reshape(-1, dim)
            posemd = pos_emd.reshape(bz, dim, -1)[:,:,self.mask.view(-1)].transpose(1, 2).reshape(-1, dim)

            if self.sigma > 0.0:
                loss_recon = self.pdist(vid_pos, posemd).mean()
                loss = torch.add(loss, self.sigma * loss_recon) # Reconstruction Loss
            
            if self.lambda_dcr > 0.0:
                loss_dcor = dcor(vid_con, vid_pos)
                if self.datasets == 'didemo':
                    loss = torch.add(loss, self.lambda_dcr * loss_dcor) # Distance correlation loss
                else:
                    loss = torch.add(loss, self.lambda_dcr * loss_dcor) # Distance correlation loss    


            regu = torch.norm(vid_con, p='fro') + torch.norm(vid_pos, p='fro')
            loss = torch.add(loss, 1e-5 * regu) # Regularization
            
            return loss 
        else:
            scores2d, _, _ = self.TCN(query, map2d_o, self.mask)
            return (scores2d).sigmoid() * self.mask
