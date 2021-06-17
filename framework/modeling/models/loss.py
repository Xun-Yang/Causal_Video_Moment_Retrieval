import torch
from torch.functional import F 

class BceLoss(object):
    def __init__(self, min_iou, max_iou, mask2d):
        self.min_iou, self.max_iou = min_iou, max_iou
        self.mask2d = mask2d
         
    def scale(self, iou):
        ious2d = (iou - self.min_iou) / (self.max_iou - self.min_iou)
        return ious2d

    def __call__(self, scores2d, ious2d):
        ious2d = self.scale(ious2d).clamp(0, 1)  

        return F.binary_cross_entropy_with_logits(
            scores2d.masked_select(self.mask2d), 
            ious2d.masked_select(self.mask2d)
        )
        
def build_bceloss(cfg, mask2d):
    min_iou = cfg.MODEL.TCN.LOSS.MIN_IOU 
    max_iou = cfg.MODEL.TCN.LOSS.MAX_IOU
    
    return BceLoss(min_iou, max_iou, mask2d) 
