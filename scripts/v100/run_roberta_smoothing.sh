model='roberta-large'
MODEL_NAME_OR_PATH="/home/xx/pretrained_model/"${model}
# dataset dir
# hyperparameter
# lr = 1e-6 get the result

epochs=12
max_seq_length=384



for((i=3;i<=7;i++));
do

for((j=1;j<=2;j++));
do
SEMEVAL_DIR_TASK2="./task2/enhanced_task2_"${j}
name=${model}_384_label_smoothing_sliding_task2_lr$i_${j}
OUTPUT_DIR=./output/${name}_lr$i_${j}
LOGGING_DIR=./logs/${name}_lr$i_${j}
learning_rate=${i}e-6
#  -m torch.distributed.launch --nproc_per_node=1  --nnodes=1\
CUDA_VISIBLE_DEVICES= python \
        run_roberta.py \
        --task_name semeval \
        --model_name_or_path ${model} \
        --label_smoothing \
        --do_train \
        --do_eval \
        --eval_all_checkpoints \
        --data_dir $SEMEVAL_DIR_TASK2 \
        --learning_rate ${learning_rate} \
        --num_train_epochs ${epochs} \
        --max_seq_length ${max_seq_length} \
        --output_dir ${OUTPUT_DIR} \
        --save_steps 40 \
        --logging_dir $LOGGING_DIR \
        --eval_steps 40 \
        --per_device_eval_batch_size=8 \
        --per_device_train_batch_size=1 \
        --gradient_accumulation_steps 16 \
        --overwrite_output  \
        --evaluate_during_training    
done
done