import json
import argparse
import numpy as np
import logging
from tqdm import tqdm
from terminaltables import AsciiTable

#from core.config import config, update_config
config_TIOU = "0.7,1.0"
config_RECALL = "1"


def iou_(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) + 1 - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) + 1 - min(pred[0], gt[0])
    return float(intersection)/union


def calculate_IoU(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))

    if union[1] - union[0] < -1e-5:
        return 0
    iou = 1.0 * (inter[1] - inter[0] + 1.0) / (union[1] - union[0] + 1.0)
    return iou if iou >= 0.0 else 0.0

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

def eval(segments, dataset, nms_thresh):
    

    result = {}

    res_iou = []
    res_rank = []
    predict_length_R1_7 = {}

    average_iou = []
    #for seg, dat in zip(segments, data):
    print("len(segments)", len(segments))

    Iou07 = 0
    for idx, seg in enumerate(segments):

        iou_re = []
        rank = []
       

        moment = dataset.get_moment(idx) 
        duration_vid = dataset.get_duration(idx) 
        sentence = dataset.get_sentence(idx) 

        seg_int = []
        for k in range(len(seg)):
            seg[k] = [int(seg[k][0]), int(seg[k][1])]

        for time in moment:
                    
            length = round((time[1]-time[0]+1)/duration_vid + 0.05, 1)
            len_id = str(length)

            if len_id not in predict_length_R1_7:            
                predict_length_R1_7[len_id] = list()
            
            Iou_score = calculate_IoU((seg[0][0] + 0.0, seg[0][1] + 0.0), (time[0] + 0.0, time[1] + 1.0))
            iou_re.append(Iou_score)
            rank.append(seg.index([time[0], time[1]+1]) + 1)
            predict_length_R1_7[len_id].append((Iou_score >= 0.7))


        res_iou.append(np.mean(np.sort(iou_re)[-3:]))
        res_rank.append(np.mean(np.sort(rank)[:3]))

    for ck in range(len(res_iou)):
        if res_iou[ck] >= 0.7:
            Iou07 += 1

    result['mIoU'] = np.mean(res_iou)
    res_rank = np.asarray(res_rank)
    result['R@1,IoU@1.0'] = np.mean(res_rank <= 1)
    result['R@1,IoU@0.7'] = Iou07/len(res_iou)
            
    eval_result = np.zeros([2,1])  
    eval_result[0,0] = result['R@1,IoU@0.7']    
    eval_result[1,0] =  result['R@1,IoU@1.0']

    miou = result['mIoU'] 
    return eval_result, miou, predict_length_R1_7


def eval_predictions(segments, dataset, nms_thresh=0.5, verbose=True):
    
    eval_result, miou, predict_length_R1_7 = eval(segments, dataset, nms_thresh)
    logger = logging.getLogger("tcn-vmr.inference")
    logger.info("Performing evaluation (Size: {}).".format(len(segments)))
    table_rank_iou = display_results(eval_result, miou, 'query-based')
    logger.info('\n' + table_rank_iou)
    if verbose == 'prior':
        print(table_rank_iou)

    # vid_recall_lan = []

    # for vid in predict_tem_lan.keys():
    #     tt = np.mean(np.array(predict_tem_lan[vid]))
    #     vid_recall_lan.append(round(tt, 3))
    #     logger.info("temporal-lang-{}: {}/({})".format(vid, round(tt, 3), len(predict_tem_lan[vid])))
    #     if verbose == 'prior':
    #         print("temporal-lang-{}: {}/({})".format(vid, round(tt, 3), len(predict_tem_lan[vid])))


    vid_recall = []

    for vid in predict_length_R1_7.keys():
        tt = np.mean(np.array(predict_length_R1_7[vid]))
        vid_recall.append(round(tt, 3))
        logger.info("length/duration-{}: {}/({})".format(vid, round(tt, 3), len(predict_length_R1_7[vid])))
        if verbose == 'prior':
            print("length/duration-{}: {}/({})".format(vid, round(tt, 3), len(predict_length_R1_7[vid])))

        
    
    vid_mIoU = np.mean(vid_recall)
    logger.info("average length-based rank1@0.7 performance: {}".format(round(vid_mIoU, 3)))
    if verbose == 'prior':
        print("average length-based rank1@0.7 performance: {}".format(round(vid_mIoU, 3)))        
        
    return eval_result, miou
    
def display_results(eval_result, miou, title=None):
    # tious = [float(i) for i in config_TIOU.split(',')] if isinstance(config_TIOU,str) else [config_TIOU]
    # print(tious)
    # recalls = [int(i) for i in config_RECALL.split(',')] if isinstance(config_RECALL,str) else [config_RECALL]
    recalls = [1]
    tious = [0.7,1.0]

    display_data = [['Rank@{},IoU@{}'.format(i,j) for i in [1] for j in [0.7,1.0]]+['mIoU']]


    eval_result = eval_result*100
    miou = miou*100
    display_data.append(['{:.02f}'.format(eval_result[j][i]) for i in range(len(recalls)) for j in range(len(tious))]+['{:.02f}'.format(miou)])

    table = AsciiTable(display_data, title)

    for i in range(len(tious)*len(recalls)):
        table.justify_columns[i] = 'center'
    return table.table
