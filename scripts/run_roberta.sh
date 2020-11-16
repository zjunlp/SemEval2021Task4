# run roberta-large on lr=1e-6 



model='roberta-large'
MODEL_NAME_OR_PATH="/home/xx/pretrained_model/"${model}
SEMEVAL_DIR_TASK1="./dataset/task1"
SEMEVAL_DIR_TASK2="./dataset/task2"
LEARNING_RATE=1e-6


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  --nnodes=1\
        run_roberta.py \
        --task_name semeval \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --do_train \
        --do_eval \
        --eval_all_checkpoints \
        --data_dir $SEMEVAL_DIR_TASK1 \
        --learning_rate $LEARNING_RATE \
        --num_train_epochs 10 \
        --max_seq_length 128 \
        --output_dir ./output/${model}_task1_128 \
        --save_steps 2000 \
        --eval_steps 500 \
        --per_device_eval_batch_size=8 \
        --per_device_train_batch_size=1 \
        --gradient_accumulation_steps 1 \
        --overwrite_output  \