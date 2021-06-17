import os
from os.path import join, dirname
import json
import logging
import pickle

import torch
import h5py
from torch.functional import F
from os.path import exists
from typing import List, Tuple
import numpy as np

from .utils import moment_to_iou2d, embedding, avgfeats, video2feats

class ActivityNetDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name, ann_file, root, feat_file, num_pre_clips, num_clips, pre_query_size):
        super(ActivityNetDataset, self).__init__()       

        with open(ann_file, 'r') as f:
            annos = json.load(f) 
            
        ann_file = ann_file.split("/")[-1]   
        self.annos = []
        self.dataset_name = dataset_name
        self.num_clips = num_clips
        self.num_pre_clips = num_pre_clips
        self.split = ann_file.split(".")[0]

        self.feat_file = feat_file

        logger = logging.getLogger("tcn-vmr.trainer")
        
        logger.info("Preparing data, please wait...")
        

        if self.split in {'train','val','test'}:
           logger.info("Loading video data, please wait...")     
           with open(self.feat_file, 'rb') as handle:                               
              self.feats = pickle.load(handle)  
           logger.info("Loading video done.")   


        if self.dataset_name in {"activitynet_ood_test_30","activitynet_ood_test_60"}:
            self.padding_num = int(self.dataset_name.split('_')[-1])
            self.rand_feats = np.load("./data/rand_feats_500.npy") 
            print('padding_time: ', self.padding_num, self.dataset_name)  
            self.vid_padding_frm = {}

        self.vids = []
        count = 0

        for vid, anno in annos.items():
            duration = anno['duration']

            if self.dataset_name in {"activitynet_ood_test_30","activitynet_ood_test_60"}:
                feat = self.feats[vid]
                frm_num = feat.shape[0]

                padding_frm_num = round((self.padding_num/duration) * frm_num)
                self.vid_padding_frm[vid] = padding_frm_num
                padding_time = padding_frm_num * (duration/frm_num)              

                duration = duration + padding_time

            # Produce annotations
            for timestamp, sentence in zip(anno['timestamps'], anno['sentences']):
                if timestamp[0] < timestamp[1]:
                    moment = [max(timestamp[0], 0), min(timestamp[1], duration)]
                else:
                    continue    

                if self.dataset_name in {"activitynet_ood_test_30","activitynet_ood_test_60"}:
                    if (moment[1] - moment[0])/anno['duration'] >= 0.5:
                        count += 1 
                        # over-length video moments, exclude from the ood test
                        continue
                    moment = [moment[0] + padding_time, moment[1] + padding_time] 

                iou2d = moment_to_iou2d(torch.tensor(moment), num_clips, duration) 
                query = embedding(sentence)  

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

        print(count)

    def __getitem__(self, idx):
        anno = self.annos[idx]
        vid = anno['vid']
        duration = anno['duration']
        moment = anno['moment']
        iou2d = anno['iou2d']

        neg_vid = self.vids[np.random.randint(len(self.vids))]
        
        while neg_vid == vid:
            neg_vid = self.vids[np.random.randint(len(self.vids))]

        feat = self.feats[vid]

        if self.dataset_name in {"activitynet_ood_test_30","activitynet_ood_test_60"}:
            padding_frm = self.vid_padding_frm[vid]

            feat = np.concatenate([self.rand_feats[0:padding_frm, :], feat], axis = 0)

        feat = torch.from_numpy(feat)
        neg_vid_fea = F.normalize(torch.from_numpy(self.feats[neg_vid]).float(), dim=1) 


        feat = F.normalize(feat, dim=1)
        feat = avgfeats(feat, self.num_pre_clips)
        neg_vid_fea  = avgfeats(neg_vid_fea, self.num_pre_clips) 

        return feat, neg_vid_fea, anno['query'], anno['wordlen'], iou2d, idx, duration
    
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

    def get_query(self, idx):
        return self.annos[idx]['query']  

    def get_iou2d(self, idx):
        return self.annos[idx]['iou2d']        

    def get_wordlen(self, idx):
        return self.annos[idx]['wordlen']                


        
