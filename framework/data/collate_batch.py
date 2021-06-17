import torch
from torch.nn.utils.rnn import pad_sequence

from framework.structures import TLGBatch


class BatchCollator(object):
    """
    Collect batch for dataloader
    """

    def __init__(self, ):
        pass

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        # [xxx, xxx, xxx], [xxx, xxx, xxx] ......
        feats, neg_vid_feats, queries, wordlens, ious2d, idxs, durations = transposed_batch
        return TLGBatch(
            feats=torch.stack(feats).float(),
            feats_neg=torch.stack(neg_vid_feats).float(),
            queries= pad_sequence(queries).transpose(0, 1),
            wordlens=torch.tensor(wordlens),
        ), torch.stack(ious2d), idxs, durations
