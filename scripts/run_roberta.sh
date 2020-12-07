model='roberta-large'
TRAIN_1_DEV_2="./dataset/train_1_dev_2"



MODEL_NAME_OR_PATH="/home/xx/pretrained_model/"${model}

# dataset dir
SEMEVAL_DIR_TASK1="./dataset/task1"
SEMEVAL_DIR_TASK2="./dataset/task2"

# hyperparameter
# lr = 1e-6 get the result

learning_rate=5e-6
epochs=8
max_seq_length=128
OUTPUT_DIR=./output/${model}_128_train_1_sliding_window

#  -m torch.distributed.launch --nproc_per_node=1  --nnodes=1\
CUDA_VISIBLE_DEVICES=2 python \
        run_roberta.py \
        --task_name semeval \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --do_train \
        --do_eval \
        --eval_all_checkpoints \
        --data_dir $SEMEVAL_DIR_TASK1 \
        --learning_rate ${learning_rate} \
        --num_train_epochs ${epochs} \
        --max_seq_length ${max_seq_length} \
        --output_dir ${OUTPUT_DIR} \
        --save_steps 2000 \
        --logging_dir $OUTPUT_DIR \
        --eval_steps 500 \
        --per_device_eval_batch_size=8 \
        --per_device_train_batch_size=1 \
        --gradient_accumulation_steps 1 \
        --overwrite_output  \
        --evaluate_during_training    \
        --fp16 
