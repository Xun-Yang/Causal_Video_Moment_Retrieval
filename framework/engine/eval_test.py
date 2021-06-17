import json
import argparse
import numpy as np
import logging
from tqdm import tqdm
from terminaltables import AsciiTable

#from core.config import config, update_config
config_TIOU = "0.3,0.5,0.7"
config_RECALL = "1,5"
def iou(pred, gt): # require pred and gt is numpy
    assert isinstance(pred, list) and isinstance(gt,list)
    pred_is_list = isinstance(pred[0],list)
    gt_is_list = isinstance(gt[0],list)
    if not pred_is_list: pred = [pred]
    if not gt_is_list: gt = [gt]
    pred, gt = np.array(pred), np.array(gt)



    inter_left = np.maximum(pred[:,0,None], gt[None,:,0])
    inter_right = np.minimum(pred[:,1,None], gt[None,:,1])
    inter = np.maximum(0.0, inter_right - inter_left)

    union_left = np.minimum(pred[:,0,None], gt[None,:,0])
    union_right = np.maximum(pred[:,1,None], gt[None,:,1])
    union = np.maximum(0.0, union_right - union_left)

    
    
    overlap = 1.0 * inter / union

    
    if not gt_is_list:
        overlap = overlap[:,0]
    if not pred_is_list:
        overlap = overlap[0]
    return overlap

def rank(pred, gt):
    return pred.index(gt) + 1

def nms(dets, thresh=0.4, top_k=-1):
    """Pure Python NMS baseline."""
    if len(dets) == 0: return []
    order = np.arange(0,len(dets),1)
    dets = np.array(dets)
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    lengths = x2 - x1
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) == top_k:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep]

def eval(segments, dataset, nms_thresh):
    method_pred = []

    tious = [float(i) for i in config_TIOU.split(',')] if isinstance(config_TIOU,str) else [config_TIOU]
    recalls = [int(i) for i in config_RECALL.split(',')] if isinstance(config_RECALL,str) else [config_RECALL]

    eval_result = [[[] for _ in recalls] for _ in tious]
    max_recall = max(recalls)
    average_iou = []
    #for seg, dat in zip(segments, data):
    print("len(segments)", len(segments))
    predict_length_R1_5 = {}

    vid_recall = []
    vid_mIoU = []   #tqdm 

    long_count = 0
    
    for idx, seg in enumerate(segments):
        seg = nms(seg, thresh=nms_thresh, top_k=max_recall).tolist()

        moment = dataset.get_moment(idx) 
        duration_vid = dataset.get_duration(idx) 
        sentence = dataset.get_sentence(idx) 
        vid_ = dataset.get_vid(idx) 
        pred_ = seg[0]



        length = round((moment[1]-moment[0])/duration_vid + 0.05, 1)
        len_id = str(length)
        
        overlap = iou(seg, [moment])
        
        average_iou.append(np.mean(np.sort(overlap[0])[-3:]))
        
        method_pred.append({
                            'idx': idx,
                            'IoU_score': round(overlap[0][0], 2),
                            'sentence': sentence,
                            'vid': vid_,
                            'moment': moment,
                            'pred': [round(pred_[0], 2), round(pred_[1], 2)],
                            'duration': duration_vid,
                        }) 

        if len_id not in predict_length_R1_5:            
            predict_length_R1_5[len_id] = list()

        for i,t in enumerate(tious):
          for j,r in enumerate(recalls):
            eval_result[i][j].append((overlap > t)[:r].any())  

        predict_length_R1_5[len_id].append((overlap > 0.5)[:1].any())            
    

    eval_result = np.array(eval_result).mean(axis=-1)
    miou = np.mean(average_iou)
 
    return eval_result, miou, predict_length_R1_5


def eval_predictions(segments, dataset, nms_thresh=0.5, verbose=True):
    
    eval_result, miou, predict_length_R1_5 = eval(segments, dataset, nms_thresh)
    logger = logging.getLogger("tcn-vmr.inference")
    logger.info("Performing evaluation (Size: {}).".format(len(segments)))
    table_rank_iou = display_results(eval_result, miou, 'query-based')
    logger.info('\n' + table_rank_iou)
    
    vid_recall = []

    for vid in predict_length_R1_5.keys():
        tt = np.mean(np.array(predict_length_R1_5[vid]))
        vid_recall.append(round(tt, 3))
        logger.info("length/duration-{}: {}/({})".format(vid, round(tt, 3), len(predict_length_R1_5[vid])))
    
    vid_mIoU = np.mean(vid_recall)
    logger.info("average length-based rank1@0.5 performance: {}".format(round(vid_mIoU, 3)))

    return eval_result, miou
    
def display_results(eval_result, miou, title=None):
    tious = [float(i) for i in config_TIOU.split(',')] if isinstance(config_TIOU,str) else [config_TIOU]
    recalls = [int(i) for i in config_RECALL.split(',')] if isinstance(config_RECALL,str) else [config_RECALL]

    display_data = [['Rank@{},mIoU@{}'.format(i,j) for i in recalls for j in tious]+['mIoU']]


    eval_result = eval_result*100
    miou = miou*100
    display_data.append(['{:.02f}'.format(eval_result[j][i]) for i in range(len(recalls)) for j in range(len(tious))]+['{:.02f}'.format(miou)])

    table = AsciiTable(display_data, title)

    for i in range(len(tious)*len(recalls)):
        table.justify_columns[i] = 'center'
    return table.table
