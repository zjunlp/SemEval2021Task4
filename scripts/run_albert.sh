model='albert-xxlarge-v2'

MODEL_NAME_OR_PATH="/home/xx/pretrained_model/"${model}


SEMEVAL_DIR_TASK1="./dataset/task1"
SEMEVAL_DIR_TASK2="./dataset/task2"

OUTPUT_DIR=./output/${model}_2021_task2
DATA_DIR=${SEMEVAL_DIR_TASK2}
# -m torch.distributed.launch --nproc_per_node=1  --nnodes=1
# lr = 1e-5 get the result

CUDA_VISIBLE_DEVICES=6 python \
        run_roberta.py \
        --task_name semeval \
        --model_name_or_path $model \
	--label_smoothing \
        --do_train \
        --do_eval \
        --eval_all_checkpoints \
        --data_dir $DATA_DIR \
        --learning_rate 3e-6 \
        --num_train_epochs 10 \
        --max_seq_length 256 \
        --output_dir  $OUTPUT_DIR \
        --logging_dir $OUTPUT_DIR \
        --save_steps 30 \
        --eval_steps 30 \
        --per_device_eval_batch_size=8 \
        --per_device_train_batch_size=1 \
        --gradient_accumulation_steps 32 \
        --evaluate_during_training  \
        --lr_scheduler linear



