# find all configs in configs/
model=cfg_16x16_pool_k5l3_activitynet

# set your gpu id
gpus=6
# number of gpus
gpun=1
# please modify it with different value (e.g., 127.0.0.2, 29502) when you run multi 2dtan task on the same machine
master_addr=127.0.0.1
master_port=29501

method=TCN_DCM

# ------------------------ need not change -----------------------------------
config_file=configs/$model\.yaml
#
for seed in {0..0}

do
    echo "training using the $seed seed\n"
    output_dir=../outputs/$model\_$method\_test_$seed\_gpu_$gpus

    CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
    --nproc_per_node=$gpun --master_addr $master_addr \
    --master_port $master_port train_net.py --config-file $config_file --seed $seed OUTPUT_DIR $output_dir SOLVER.MAX_EPOCH 20 SOLVER.LR 0.0001 SOLVER.BATCH_SIZE 64 TEST.BATCH_SIZE 64 MODEL.ETA 1.0 MODEL.LAMBDA 0.001 MODEL.ARCHITECTURE $method 
done

