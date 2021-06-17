import argparse
import os

import torch
import numpy as np

from framework.config import cfg
from framework.data import make_data_loader
from framework.engine.inference import inference
from framework.modeling import build_model
from framework.utils.checkpoint import TcnCheckpointer
from framework.utils.comm import synchronize, get_rank
from framework.utils.logger import setup_logger

def main(args):


    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("tcn-vmr", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)
    
    model = build_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = TcnCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)



    data_loaders_test_IID = make_data_loader(cfg, is_train=2, is_distributed=distributed, is_for_period=False, numworkers=4)
    data_loaders_test_OOD_1 = make_data_loader(cfg, is_train=3, is_distributed=distributed, is_for_period=False, numworkers=4)
    data_loaders_test_OOD_2 = make_data_loader(cfg, is_train=4, is_distributed=distributed, is_for_period=False, numworkers=4)


    for data_loader_iid_test, data_loader_1_ood_test, data_loader_2_ood_test in zip(data_loaders_test_IID, data_loaders_test_OOD_1, data_loaders_test_OOD_2):
        inference(
            model,
            data_loader_iid_test,
            dataset_name=cfg.DATASETS.TEST,
            nms_thresh=cfg.TEST.NMS_THRESH,
            num_clips=cfg.MODEL.TCN.NUM_CLIPS,
            device=cfg.MODEL.DEVICE,
        )
        inference(
            model,
            data_loader_1_ood_test,
            dataset_name=cfg.DATASETS.TEST_1,
            nms_thresh=cfg.TEST.NMS_THRESH,
            num_clips=cfg.MODEL.TCN.NUM_CLIPS,
            device=cfg.MODEL.DEVICE,
        )
        inference(
            model,
            data_loader_2_ood_test,
            dataset_name=cfg.DATASETS.TEST_2,
            nms_thresh=cfg.TEST.NMS_THRESH,
            num_clips=cfg.MODEL.TCN.NUM_CLIPS,
            device=cfg.MODEL.DEVICE,
        )                            
        synchronize()
 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tcn")
    parser.add_argument(
        "--config-file",
        default="configs/cfg_16x16_pool_k5l8_charades_I3D.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--seed",
        default=0,
        metavar="seed",
        help="random seed",
        type=int,
    )

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
    main(args)
