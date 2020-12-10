model="spanbert-large-cased"

MODEL_NAME_OR_PATH="/home/xx/pretrained_model/"${model}


SEMEVAL_DIR="./dataset/task1"
SEMEVAL_DIR_TASK2="./dataset/task2"

OUTPUT_DIR=./output/${model}_128_task1_test

# -m torch.distributed.launch --nproc_per_node=1  --nnodes=1

CUDA_VISIBLE_DEVICES=2 python \
        run_spanbert.py \
        --task_name semeval \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --do_train \
        --do_eval \
        --eval_all_checkpoints \
        --data_dir $SEMEVAL_DIR \
        --learning_rate 9e-6\
        --num_train_epochs 10 \
        --max_seq_length 128 \
        --output_dir ${OUTPUT_DIR} \
        --logging_dir ${OUTPUT_DIR} \
        --save_steps 500 \
        --eval_steps 500 \
        --per_device_eval_batch_size=1 \
        --per_device_train_batch_size=1 \
        --gradient_accumulation_steps 1 \
        --evaluate_during_training   \
        --overwrite_cache \
        --overwrite_output_dir  \











# baseline
# CUDA_VISIBLE_DEVICES=2 python ./Baselines/Run_GAReader.py