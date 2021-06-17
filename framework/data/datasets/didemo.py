import os
from os.path import join, dirname, exists

import json
import h5py
import logging
import numpy as np
import torch.nn.functional as F


import torch

from .utils import video2feats, moment_to_iou2d_didemo, embedding, avgfeats

class DiDeMoDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name, root, ann_file, feat_file, num_pre_clips=6, num_clips=6, pre_query_size=0):
        super(DiDeMoDataset, self).__init__()
        with open(ann_file,'r') as f:
            annos = json.load(f)
  

        self.dataset_name = dataset_name
        self.feat_flow_file = feat_file

        self.num_clips = 6
        self.num_pre_clips = 6

        if self.dataset_name in {'didemo_ood_test_3','didemo_ood_test_6'}:
            
            self.padding_clips = int(self.dataset_name.split('_')[-1])
            self.rand_feats = np.load("./data/rand_feats_1024.npy")

            print(self.dataset_name)
            print('padding_clips: ', self.padding_clips)
            self.padding_frm = int(self.padding_clips)

        self.annos =  []
        self.vids = []


        logger = logging.getLogger("tcn-vmr.trainer")
        logger.info("Preparing data, please wait...")           

        self.location = {}
        for d in annos:
            vid = d['video']
            sentence = d['description']
            times = d['times']
            
            duration = 6.0
            moment = times

            
            query = embedding(sentence)

            if self.dataset_name in {'didemo_ood_test_3','didemo_ood_test_6'}:
                                
                duration = 6.0 + self.padding_frm

                if isinstance(times[0], list):
                    for i in range(len(times)):
                        moment[i][0] = moment[i][0] + self.padding_frm
                        moment[i][1] = moment[i][1] + self.padding_frm
                else:    
                    moment[0] = moment[0] + self.padding_frm
                    moment[1] = moment[1] + self.padding_frm

            if isinstance(times[0],list):
                rint = np.random.randint(0, len(times))
                moment_ = [times[rint][0], times[rint][1]]
            else:
                moment_ = times

            iou2d = moment_to_iou2d_didemo(torch.tensor(moment_), 6, 6.0)


            self.vids.append(vid)                            
            self.annos.append(
                {
                    'vid': vid,
                    'moment': moment,
                    'iou2d': iou2d, 
                    'sentence': sentence,
                    'query': query,
                    'wordlen': query.size(0),
                    'duration': duration,
                }
            )
        
        self.feats_flow = {}
        with h5py.File(self.feat_flow_file, 'r') as f:
            for key in f.keys():
                self.feats_flow[key] = f[key][:]
  

    def __getitem__(self, idx):
        anno = self.annos[idx]
        vid = anno['vid']

        neg_vid = self.vids[np.random.randint(len(self.vids))]
        
        while neg_vid == vid:
            neg_vid = self.vids[np.random.randint(len(self.vids))]

        visual_input = self.feats_flow[vid]
        neg_vid_fea = F.normalize(torch.from_numpy(self.feats_flow[neg_vid]).float(), dim=1) 

        assert visual_input.shape[0] <= 6
        assert neg_vid_fea.shape[0] <= 6

        if self.dataset_name in {'didemo_ood_test_3','didemo_ood_test_6'}:

            rand_fea_pad = self.rand_feats[0: self.padding_frm, :]
            visual_input = np.concatenate([rand_fea_pad, visual_input], axis = 0)


        feat = torch.from_numpy(visual_input).float()
        feat = F.normalize(feat, dim=1)

        return feat, neg_vid_fea, anno['query'], anno['wordlen'], anno['iou2d'], idx, anno['duration']

    def __len__(self):
        return len(self.annos)

    def get_duration(self, idx):
        return self.annos[idx]['duration']

    def get_sentence(self, idx):
        return self.annos[idx]['sentence']

    def get_moment(self, idx):
        return self.annos[idx]['moment']

    def get_vid(self, idx):
        return self.annos[idx]['vid']
