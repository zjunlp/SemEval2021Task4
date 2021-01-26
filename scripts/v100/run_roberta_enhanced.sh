model='roberta-large'
MODEL_NAME_OR_PATH="./pretrained_model/"${model}
# dataset dir
# hyperparameter
# lr = 1e-6 get the result

epochs=12
max_seq_length=256

for((i=3;i<=6;i++))
do
for((j=1;j<=2;j++))
do
learning_rate=${i}e-6
OUTPUT_DIR=./output/${model}_256_enhanced_${i}
LOGGING_DIR=./logs/${model}_256_enhanced_${i}
DATA_DIR="./task2/enhanced_task2_"${j}
#  -m torch.distributed.launch --nproc_per_node=1  --nnodes=1\
CUDA_VISIBLE_DEVICES= python \
        run_roberta.py \
        --task_name semeval \
        --model_name_or_path ${model} \
        --do_train \
        --do_eval \
        --label_smoothing \
        --eval_all_checkpoints \
        --data_dir $DATA_DIR \
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
        --evaluate_during_training    \
        --overwrite_cache
done
done

