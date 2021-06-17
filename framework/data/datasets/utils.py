import os
from os.path import join, exists
import h5py
import numpy as np

import torch
import torchtext
from torch.functional import F


import json
import yaml
import csv
import logging

from tqdm import tqdm

from collections import defaultdict, OrderedDict
import string


def iou(candidates, gt):
    start, end = candidates[:,0].float(), candidates[:,1].float()
    s, e = gt[0].float(), gt[1].float()
    # print(s.dtype, start.dtype)
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union

def iou_didemo(candidates, gt):

    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0].float(), (gt[1]+1.0).float()
    # print(s.dtype, start.dtype)
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union

def score2d_to_moments_scores(score2d, num_clips, duration):
    grids = torch.nonzero(score2d, as_tuple=False)#.nonzero()   
    scores = score2d[grids[:,0], grids[:,1]]
    grids[:, 1] += 1
    moments = grids * duration / num_clips
    return moments, scores

def moment_to_iou2d(moment, num_clips, duration):
    iou2d = torch.ones(num_clips, num_clips)
    candidates, _ = score2d_to_moments_scores(iou2d, num_clips, duration)
    iou2d = iou(candidates, moment).reshape(num_clips, num_clips)
    return iou2d

def moment_to_iou2d_didemo(moment, num_clips, duration):

    iou2d = torch.ones(num_clips, num_clips)
    candidates, _ = score2d_to_moments_scores(iou2d, num_clips, torch.tensor(duration).float())
    

    iou2d = iou_didemo(candidates, moment).reshape(num_clips, num_clips)


    return iou2d


def avgfeats(feats, num_pre_clips):
    # Produce the feature of per video into fixed shape (e.g. 256*4096)
    # Input Example: feats (torch.tensor, ?x4096); num_pre_clips (256)
    num_src_clips = feats.size(0)
    idxs = torch.arange(0, num_pre_clips+1, 1.0) / num_pre_clips * num_src_clips
    idxs = idxs.round().long().clamp(max=num_src_clips-1)
    # To prevent a empty selection, check the idxs
    meanfeats = []
    for i in range(num_pre_clips):
        s, e = idxs[i], idxs[i+1]
        if s < e:
            meanfeats.append(feats[s:e].mean(dim=0))
        else:
            meanfeats.append(feats[s])
    return torch.stack(meanfeats)
    
def video2feats(feat_file, vids, num_pre_clips, dataset_name):
    assert exists(feat_file)
    vid_feats = {}
    with h5py.File(feat_file, 'r') as f:
        for vid in vids:
            if dataset_name == "activitynet":
                feat = f[vid]['c3d_features'][:]
                #feat = F.normalize(torch.from_numpy(feat),dim=1)
                vid_feats[vid] = feat
            else:
                feat = f[vid][:]
                #feat = F.normalize(torch.from_numpy(feat),dim=1)
                vid_feats[vid] = feat#avgfeats(feat, num_pre_clips) 
    return vid_feats

def embedding(sentence, vocabs=[], embedders=[]):
    if len(vocabs) == 0:
        vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
        vocab.itos.extend(['<unk>'])
        vocab.stoi['<unk>'] = vocab.vectors.shape[0]
        vocab.vectors = torch.cat(
            [vocab.vectors, torch.zeros(1, vocab.dim)],
            dim=0
        )
        vocabs.append(vocab)
    
    if len(embedders) == 0:
        embedder = torch.nn.Embedding.from_pretrained(vocab.vectors)
        embedders.append(embedder)
    
    vocab, embedder = vocabs[0], embedders[0]

    if isinstance(sentence, list):
        word_idxs = torch.tensor([vocab.stoi.get(w.lower(), 400000) \
            for w in sentence], dtype=torch.long)
    else:        
        word_idxs = torch.tensor([vocab.stoi.get(w.lower(), 400000) \
            for w in sentence.split()], dtype=torch.long)
    return embedder(word_idxs)

""" Methods for string """
def tokenize(txt, translator=None):
    """ Tokenize text - converting to lower characters, eliminating puncutations
    Args:
        txt: text to be tokenized; str
        translator: this includes punctuations for translate() func; dict()
    """
    if not translator:
        translator = str.maketrans("", "", string.punctuation)
    tokens = str(txt).lower().translate(translator).strip().split()
    return tokens


def check_and_create_dir(dir_path):
    if not os.path.isdir(dir_path):
        print("Create directory: {}".format(dir_path))
        os.makedirs(dir_path, exist_ok=True)

def load_csv(file_path):
    out = []
    with open(file_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(row)
    return out

def load_json(file_path, verbose=True):
    if verbose:
        print("Load json file from {}".format(file_path))
    return json.load(open(file_path, "r"))

""" HDF5 helpers """
def open_hdf5(file_path, mode="r", verbose=True):
    if verbose:
        print("Open hdf5 file from {}".format(file_path))
    return h5py.File(file_path, mode)

def load_hdf5(file_path, verbose=True):
    if verbose:
        print("Load hdf5 file from {}".format(file_path))
    return h5py.File(file_path, "r")

def load_lines_from(file_path):
    lines = []
    f = open(file_path, "r")
    while True:
        line = f.readline()
        if not line: break
        lines.append(line.strip().strip("\n"))
    f.close()
    return lines