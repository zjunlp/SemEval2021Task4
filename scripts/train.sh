model='roberta-large'
MODEL_NAME_OR_PATH="/home/xx/pretrained_model/"${model}
SEMEVAL_DIR_TASK2="./dataset/enhanced_roberta_task2"

TASK_NAME="semevalenhanced"
OUTPUT_DIR=./output/${model}_quiantao

# enhanced on task 1  hope it will work on roberta-large

# sliding window, label smooth, enhanced, cos



CUDA_VISIBLE_DEVICES=3 python    run_roberta.py \
        --task_name $TASK_NAME \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --do_train \
        --label_smoothing \
        --sliding \
        --do_eval \
        --data_dir $SEMEVAL_DIR_TASK2 \
        --learning_rate 5e-6 \
        --num_train_epochs 3 \
        --max_seq_length 256 \
        --output_dir ${OUTPUT_DIR} \
        --save_steps 50 \
        --eval_steps 50 \
        --per_device_eval_batch_size=8 \
        --per_device_train_batch_size=1 \
        --gradient_accumulation_steps 1 \
        --logging_dir $OUTPUT_DIR \
        --overwrite_output_dir \
        --gradient_accumulation_steps 16 \
        --evaluate_during_training  