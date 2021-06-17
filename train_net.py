import argparse
import os

import torch
from torch import optim
from torch import multiprocessing
import numpy as np
multiprocessing.set_sharing_strategy('file_system')

from framework.config import cfg
from framework.data import make_data_loader
from framework.engine.inference import inference
from framework.engine.trainer import do_train
from framework.modeling import build_model
from framework.utils.checkpoint import TcnCheckpointer
from framework.utils.comm import synchronize, get_rank
from framework.utils.imports import import_file
from framework.utils.logger import setup_logger
from framework.utils.miscellaneous import mkdir, save_config

def train(cfg, local_rank, distributed):

    
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)

    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.MILESTONES, gamma=0.1)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            #broadcast_buffers=False,
        )

    output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0
    checkpointer = TcnCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )

    if cfg.MODEL.WEIGHT != "":
        extra_checkpoint_data = checkpointer.load(f=None, use_latest=True)
    else:
        extra_checkpoint_data = checkpointer.load(f=cfg.MODEL.WEIGHT, use_latest=False)
    
    arguments = {"epoch": 1}
    arguments.update(extra_checkpoint_data)
    
    data_loader = make_data_loader(
        cfg,
        is_train=0, 
        is_distributed=distributed, 
        is_for_period=True, 
        numworkers=4
    )

    test_period = cfg.SOLVER.TEST_PERIOD
    if test_period > 0:
        # original testing set
        data_loader_val = make_data_loader(cfg, is_train=2, is_distributed=distributed, is_for_period=True, numworkers=4) 
        data_loader_test = None 
    else:
        data_loader_val = None
        data_loader_test= None

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    
    do_train(
        cfg,
        model,
        data_loader,
        data_loader_val,
        data_loader_test,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        test_period,
        arguments,
    )

    return model

def run_test(cfg, model, distributed):#best_epoch
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps

    

    output_dir = cfg.OUTPUT_DIR
    checkpointer = TcnCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = output_dir + "/model_best.pth"#.format(best_epoch)#cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=False)
    print("\n Testing using the model checkpoint\n")


    # dataset_names = cfg.DATASETS.TEST
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

def main(args):
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1


    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("tcn-vmr", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)
    model = train(cfg, args.local_rank, args.distributed)           

    if not args.skip_test:
        run_test(cfg, model, args.distributed)
        exit()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tcn")
    parser.add_argument(
        "--config-file",
        default="configs/cfg_16x16_pool_k5l8_charades_I3D.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
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
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
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
    torch.cuda.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.set_num_threads(6)
    
    main(args)
