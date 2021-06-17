import torch
from torch import nn
from torch.functional import F

class Integrator(nn.Module):
    def __init__(self, feat_hidden_size, query_input_size, query_hidden_size, 
            bidirectional, num_layers):
            
        super(Integrator, self).__init__()
        if bidirectional:
            query_hidden_size //= 2

        self.fc = nn.Linear(query_hidden_size, feat_hidden_size)
        self.conv = nn.Conv2d(feat_hidden_size, feat_hidden_size, 1, 1)



    def forward(self, queries, wordlens, map2d):
                
        queries = self.fc(queries)
        queries = queries[:, :, None, None]

        map2d = self.conv(map2d)

        return queries, map2d
        

def build_integrator(cfg):
    feat_hidden_size = cfg.MODEL.TCN.FEATPOOL.HIDDEN_SIZE 
    query_input_size = cfg.INPUT.PRE_QUERY_SIZE
    query_hidden_size = cfg.MODEL.TCN.INTEGRATOR.QUERY_HIDDEN_SIZE

    bidirectional = cfg.MODEL.TCN.INTEGRATOR.LSTM.BIDIRECTIONAL
    num_layers = cfg.MODEL.TCN.INTEGRATOR.LSTM.NUM_LAYERS


    return Integrator(feat_hidden_size, query_input_size, query_hidden_size, 
        bidirectional, num_layers) 
