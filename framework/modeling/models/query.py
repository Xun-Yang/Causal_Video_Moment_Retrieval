import torch
from torch import nn
from torch.functional import F

class Integrator(nn.Module):
    def __init__(self, feat_hidden_size, query_input_size, query_hidden_size, 
            bidirectional, num_layers):
        super(Integrator, self).__init__()


        if bidirectional:
            query_hidden_size //= 2

        self.lstm = nn.LSTM(
            query_input_size, query_hidden_size, num_layers=num_layers, 
            bidirectional=bidirectional, batch_first=True
        )


    def encode_query(self, queries, wordlens):

        self.lstm.flatten_parameters()
        queries = self.lstm(queries)[0]

        queries_last = queries[range(queries.size(0)), wordlens.long() - 1]
        queries_max = torch.stack([queries[i,:wordlens[i].long()].max(-2)[0] for i in range(queries.size(0))], dim=0)

        return queries_last, queries_max

    def forward(self, queries, wordlens):
                
        queries = self.encode_query(queries, wordlens)

        return queries


def build_query(cfg):

    feat_hidden_size = cfg.MODEL.TCN.FEATPOOL.HIDDEN_SIZE 
    query_input_size = cfg.INPUT.PRE_QUERY_SIZE
    query_hidden_size = cfg.MODEL.TCN.INTEGRATOR.QUERY_HIDDEN_SIZE 

    bidirectional = cfg.MODEL.TCN.INTEGRATOR.LSTM.BIDIRECTIONAL
    num_layers = cfg.MODEL.TCN.INTEGRATOR.LSTM.NUM_LAYERS

    return Integrator(feat_hidden_size, query_input_size, query_hidden_size, bidirectional, num_layers) 
