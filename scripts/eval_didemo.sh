# find all configs in configs/

model=cfg_6x6_pool_k5l3_didemo

gpus=6
# number of gpus
gpun=1
# please modify it with different value (e.g., 127.0.0.2, 29502) when you run multi task on the same machine
master_addr=127.0.0.1
master_port=29501

method=TCN_DCM

seed=1

config_file=configs/$model\.yaml

for seed in {0..0}
do 
    echo "evaluate using the $seed seed model weight\n"
    output_dir=./outputs/$model\_$method\_test_$seed\_gpu_$gpus
    weight_file=$output_dir/model_best.pth

    CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
    --nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port \
    test_net.py --config-file $config_file --ckpt $weight_file --seed $seed OUTPUT_DIR $output_dir MODEL.ARCHITECTURE $method SOLVER.LR 0.0001 SOLVER.BATCH_SIZE 64 TEST.BATCH_SIZE 64

done
