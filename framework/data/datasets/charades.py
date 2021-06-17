import os
from os.path import join, dirname
import json
import logging
import csv

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import torchtext
from os.path import exists
import pickle
from .utils import video2feats, moment_to_iou2d, embedding, avgfeats

class CharadesDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name, root, ann_file, feat_file, num_pre_clips, num_clips, pre_query_size):
        super(CharadesDataset, self).__init__()
        self.data_dir = root
        self.dataset_name = dataset_name
        self.split = ann_file.split('_')[-1].split('.')[0]
        self.num_pre_clips = num_pre_clips

        self.durations = {}
        self.feat_type = self.dataset_name.split('_')[1]
        print("use feature: {}".format(self.feat_type))
        self.feat_path = feat_file + "/{}.npy"   


        with open(os.path.join(self.data_dir, 'Charades_v1_{}.csv'.format(self.split))) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.durations[row['id']] = float(row['length']) 


        anno_file = open(os.path.join(self.data_dir, "charades_sta_{}.txt".format(self.split)),'r')
        if self.dataset_name in {"charades_VGG_ood_test_10", "charades_VGG_ood_test_15", "charades_C3D_ood_test_10", "charades_C3D_ood_test_15", "charades_I3D_ood_test_10", "charades_I3D_ood_test_15"}:
            self.padding_time = int(self.dataset_name.split('_')[-1])

            if self.feat_type == 'VGG':
                self.rand_feats = np.load("./data/rand_feats_4096.npy") 
            elif self.feat_type == 'C3D' or self.feat_type == 'I3D':
                self.rand_feats = np.load("./data/rand_feats_1024.npy")       

            print('padding_time: ', self.padding_time, self.dataset_name)            
            self.vid_padding_frm = {}



        self.annos =  []
        self.vids = []
        logger = logging.getLogger("tcn-vmr.trainer")
        logger.info("Preparing data, please wait...")
        

        for line in anno_file:
            anno, sent = line.split("##")
            sent = sent.split('.\n')[0]
            vid, s_time, e_time = anno.split(" ")
            s_time = float(s_time)
            e_time = min(float(e_time), self.durations[vid])
            duration = self.durations[vid]

            if self.dataset_name in {"charades_VGG_ood_test_10", "charades_VGG_ood_test_15", "charades_C3D_ood_test_10", "charades_C3D_ood_test_15", "charades_I3D_ood_test_10", "charades_I3D_ood_test_15"}:
                features =  np.load(self.feat_path.format(vid))
                frm_num = features.shape[0]

                features = []
                padding_frm = round(self.padding_time * float(frm_num)/duration)

                self.vid_padding_frm[vid] = padding_frm

                padding_time = float(format(float(padding_frm) * duration/float(frm_num), '.2f'))#
                
                duration = duration + padding_time

                assert padding_frm < 200
                
                s_time = s_time + padding_time
                e_time = e_time + padding_time  
                              
            if s_time < e_time:
                
                moment = [max(s_time, 0), e_time]
                iou2d = moment_to_iou2d(torch.tensor(moment), num_clips, duration)
                query = embedding(sent)  
                
                self.vids.append(vid)
                self.annos.append(
                        {
                            'vid': vid,
                            'moment': moment,
                            'iou2d': iou2d, 
                            'sentence': sent,
                            'query': query,
                            'wordlen': query.size(0),
                            'duration': duration,
                        }
                )
        anno_file.close()         

    def __getitem__(self, idx):
        anno = self.annos[idx]
        vid = anno['vid']
        neg_vid = self.vids[np.random.randint(len(self.vids))]
        
        while neg_vid == vid:
            neg_vid = self.vids[np.random.randint(len(self.vids))]


        features =  np.load(self.feat_path.format(vid))

        if len(features.shape) == 4:
            features = np.squeeze(features, axis=1)
            features = np.squeeze(features, axis=1)

        if self.feat_type == 'C3D':
            assert features.shape[-1] == 1024

        neg_vid_fea = np.load(self.feat_path.format(neg_vid))


        if len(neg_vid_fea.shape) == 4:
            neg_vid_fea = np.squeeze(neg_vid_fea, axis=1)
            neg_vid_fea = np.squeeze(neg_vid_fea, axis=1)

        neg_vid_fea = F.normalize(torch.from_numpy(neg_vid_fea).float(), dim=1)

        if self.dataset_name in {"charades_VGG_ood_test_10", "charades_VGG_ood_test_15", "charades_C3D_ood_test_10", "charades_C3D_ood_test_15", "charades_I3D_ood_test_10", "charades_I3D_ood_test_15"}:
            padding_frm = self.vid_padding_frm[vid]#round(self.padding_time * float(frm_num)/self.durations[video_id])
            
            assert padding_frm < 200 and padding_frm >= 1

            features = np.concatenate([self.rand_feats[0:padding_frm, :], features], axis = 0)


        feat = F.normalize(torch.from_numpy(features).float(), dim=1)    

        feat = avgfeats(feat, self.num_pre_clips) 
        neg_vid_fea  = avgfeats(neg_vid_fea, self.num_pre_clips) 

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


if __name__ == '__main__':
    print('testing')