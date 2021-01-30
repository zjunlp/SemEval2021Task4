model='roberta-large'
MODEL_NAME_OR_PATH="./pretrained_model/"${model}
enhanced_model_path="output/pretrain_test3/roberta-large"
# dataset dir
SEMEVAL_DIR_TASK1="./dataset/task1"
SEMEVAL_DIR_TASK2="./task2/enhanced_task2_1"
# hyperparameter
# lr = 1e-6 get the result

epochs=12
max_seq_length=256

for((i=1;i<=5;i++))
do
learning_rate=${i}e-5
OUTPUT_DIR=./output/${model}_256_smoothing_lr${i}e-5
LOGGING_DIR=./logs/${model}_256_smoothing_lr${i}e-5
#  -m torch.distributed.launch --nproc_per_node=1  --nnodes=1\
CUDA_VISIBLE_DEVICES=2 python \
        run_roberta.py \
        --task_name semeval \
        --model_name_or_path ${enhanced_model_path} \
        --do_train \
        --do_eval \
        --label_smoothing \
        --eval_all_checkpoints \
        --data_dir $SEMEVAL_DIR_TASK2 \
        --learning_rate ${learning_rate} \
        --num_train_epochs ${epochs} \
        --max_seq_length ${max_seq_length} \
        --output_dir ${OUTPUT_DIR} \
        --logging_step 20 \
        --save_steps 20 \
        --logging_dir $LOGGING_DIR \
        --eval_steps 20 \
        --per_device_eval_batch_size=8 \
        --per_device_train_batch_size=1 \
        --gradient_accumulation_steps 32 \
        --overwrite_output  \
        --evaluate_during_training    \
        --overwrite_cache

done
