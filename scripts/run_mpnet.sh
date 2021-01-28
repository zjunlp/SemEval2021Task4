model='mpnet-base'
MODEL_NAME_OR_PATH="/home/xx/pretrained_model/"${model}
# dataset dir
SEMEVAL_DIR_TASK2="./dataset/task2"
# hyperparameter
# lr = 1e-6 get the result

epochs=12
max_seq_length=256

for((i=3;i<=9;i++))
do
learning_rate=${i}e-6
OUTPUT_DIR=./output/${model}_256_smoothing_lr${i}
LOGGING_DIR=./logs/${model}_256_smoothing_lr${i}
#  -m torch.distributed.launch --nproc_per_node=1  --nnodes=1\
CUDA_VISIBLE_DEVICES=2 python \
        run_roberta.py \
        --task_name semeval \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --do_train \
        --do_eval \
        --label_smoothing \
        --eval_all_checkpoints \
        --data_dir $SEMEVAL_DIR_TASK2 \
        --learning_rate ${learning_rate} \
        --num_train_epochs ${epochs} \
        --max_seq_length ${max_seq_length} \
        --output_dir ${OUTPUT_DIR} \
        --save_steps 20 \
        --logging_dir $LOGGING_DIR \
        --eval_steps 20 \
        --per_device_eval_batch_size=8 \
        --per_device_train_batch_size=1 \
        --gradient_accumulation_steps 32 \
        --overwrite_output  \
        --evaluate_during_training    
        # --overwrite_cache

done
