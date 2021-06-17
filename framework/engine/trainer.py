import datetime
import logging
import os
import time

import torch
import torch.distributed as dist

from framework.data import make_data_loader
from framework.utils.comm import get_world_size, synchronize, is_main_process
from framework.utils.metric_logger import MetricLogger
from framework.engine.inference import inference

def reduce_loss(loss):
    world_size = get_world_size()
    if world_size < 2:
        return loss
    with torch.no_grad():
        dist.reduce(loss, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            loss /= world_size
    return loss

def do_train(
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
):
    logger = logging.getLogger("tcn-vmr.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_epoch = cfg.SOLVER.MAX_EPOCH

    model.train()
    start_training_time = time.time()
    end = time.time()
    best_performance_epoch = [0.0, 0]

    for epoch in range(arguments["epoch"], max_epoch + 1):
        max_iteration = len(data_loader)
        last_epoch_iteration = (max_epoch - epoch) * max_iteration
        arguments["epoch"] = epoch

        for iteration, (batches, targets, _, _) in enumerate(data_loader):
            iteration += 1
            data_time = time.time() - end

            batches = batches.to(device)
            targets = targets.to(device)
            
            def closure():
                optimizer.zero_grad()
                loss = model(batches, targets, model.training)
                if iteration % 20 == 0 or iteration == max_iteration:
                    meters.update(loss=reduce_loss(loss.detach()))
                loss.backward()
                return loss

            optimizer.step(closure)

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iteration - iteration + last_epoch_iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 40 == 0 or iteration == max_iteration:
                logger.info(
                    meters.delimiter.join(
                        [
                            #"eta: {eta}",
                            "epoch: {epoch}/{max_epoch}",
                            "iteration: {iteration}/{max_iteration}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            #"max mem: {memory:.0f}",
                        ]
                    ).format(
                        #eta=eta_string,
                        epoch=epoch,
                        max_epoch=max_epoch,
                        iteration=iteration,
                        max_iteration=max_iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        #memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )

        # scheduler.step()
        # if epoch % checkpoint_period == 0:
        #     checkpointer.save(f"model_{epoch}e", **arguments)

        if data_loader_val is not None and test_period > 0 and \
            epoch % test_period == 0:
            synchronize()
            result = inference(
                model,
                data_loader_val,
                dataset_name=cfg.DATASETS.VAL,
                nms_thresh=cfg.TEST.NMS_THRESH,
                num_clips=cfg.MODEL.TCN.NUM_CLIPS,                
                device="cuda"
            )
            

            if result is not None:
                dataname = cfg.DATASETS.VAL[0].split('_')[0]
                eval_result, miou = result 
                if dataname == 'didemo':
                    performance = (eval_result[0][0] + eval_result[1][0])*100
                    if performance > best_performance_epoch[0]:
                        best_performance_epoch[0] = performance
                        best_performance_epoch[1] = epoch
                        logger.info("The current best epoch is epoch_{}: Rank1@0.7-{:.02f}, Rank1@1.0-{:.02f}, mIoU-{:.02f}".format(epoch, eval_result[0][0]*100, eval_result[1][0]*100, miou*100))
                        checkpointer.save("model_best")                    
                else:    

                    performance = (eval_result[1][0] + eval_result[2][0])*100
                    if performance > best_performance_epoch[0]:
                        best_performance_epoch[0] = performance
                        best_performance_epoch[1] = epoch
                        logger.info("The current best epoch (R1-0.5+R1-0.7) is epoch_{}: Rank1@0.5-{:.02f}, Rank1@0.7-{:.02f}, mIoU-{:.02f}".format(epoch, eval_result[1][0]*100, eval_result[2][0]*100, miou*100))
                        checkpointer.save("model_best")

            synchronize() 
            model.train()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iteration)
        )
    )
    #return best_performance_epoch[1]