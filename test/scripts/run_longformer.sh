model='longformer-base-4096'



MODEL_NAME_OR_PATH="/home/xx/pretrained_model/"${model}


SEMEVAL_DIR="./dataset/task1"
SEMEVAL_DIR_TASK2="./dataset/task2"
TRAIN_1_DEV_2="./dataset/train_1_dev_2"

# -m torch.distributed.launch --nproc_per_node=1  --nnodes=1

CUDA_VISIBLE_DEVICES=2 python \
        run_roberta.py \
        --task_name semeval \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --do_train \
        --do_eval \
        --eval_all_checkpoints \
        --data_dir $SEMEVAL_DIR_TASK2 \
        --learning_rate 1e-5 \
        --num_train_epochs 10 \
        --max_seq_length 1024 \
        --output_dir ./output/${model}_task2_128 \
        --save_steps 2000 \
        --fp16 \
        --per_device_eval_batch_size=8 \
        --per_device_train_batch_size=1 \
        --gradient_accumulation_steps=32 \
        --evaluate_during_training   \
        --overwrite_output_dir











# baseline
# CUDA_VISIBLE_DEVICES=2 python ./Baselines/Run_GAReader.py