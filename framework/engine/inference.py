import logging
import time
import os
import numpy as np
from tqdm import tqdm

import torch

from framework.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from . import eval_test, eval_test_didemo

def compute_on_dataset(model, data_loader, num_clips, device, timer=None, dataset_name=None):
    model.eval()
    results_dict = {}
    dcc_value = []

    cpu_device = torch.device("cpu")

    for batch in tqdm(data_loader):
        batches, _, idxs, durations = batch

        with torch.no_grad():
            if timer:
                timer.tic()
            output = model(batches.to(device), None, False)

            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()

            if dataset_name == 'didemo' and batches.feats.shape[-2] > num_clips:
                num_clips = batches.feats.shape[-2]

            sorted_times = get_proposal_results(output, durations, num_clips)

        results_dict.update(
            {img_id: result for img_id, result in zip(idxs, sorted_times)}
        )   
    return results_dict


def get_proposal_results(scores, durations, num_clips):
        # assume all valid scores are larger than one
        out_sorted_times = []
        for score, duration in zip(scores, durations):
            T = score.shape[-1]
            sorted_indexs = np.dstack(np.unravel_index(np.argsort(score.cpu().detach().numpy().ravel())[::-1], (T, T))).tolist()

            sorted_indexs = np.array([item for item in sorted_indexs[0] if item[0] <= item[1]]).astype(float)

            sorted_indexs[:,1] = sorted_indexs[:,1] + 1
            
            sorted_indexs = torch.from_numpy(sorted_indexs).cuda()
            target_size = num_clips#config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
            out_sorted_times.append((sorted_indexs.float() / target_size * duration).tolist())

        return out_sorted_times


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)

    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    idxs = list(sorted(predictions.keys()))


    if len(idxs) != idxs[-1] + 1:
        logger = logging.getLogger("tcn-vmr.inference")
        logger.warning(
            "Number of samples that were gathered from multiple processes is not "
            "a contiguous set. Some samples might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in idxs]
    
    return predictions

def inference(
        model,
        data_loader,
        dataset_name,
        nms_thresh,
        num_clips,
        device="cuda"):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("tcn-vmr.inference")
    dataset = data_loader.dataset

    datasets_name = dataset_name[0].split('_')[0]

    logger.info("Start evaluation on {} dataset (Size: {}).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, num_clips, device, inference_timer, datasets_name)
    
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / inference per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / inference per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return
    
    if datasets_name == 'didemo':
        return eval_test_didemo.eval_predictions(predictions, dataset=dataset, nms_thresh=nms_thresh, verbose=True)
    else:    
        return eval_test.eval_predictions(predictions, dataset=dataset, nms_thresh=nms_thresh, verbose=True)
