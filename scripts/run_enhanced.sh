model='roberta-large'
MODEL_NAME_OR_PATH="/home/xx/pretrained_model/"${model}
SEMEVAL_DIR_TASK1="./dataset/enhanced_roberta_task1"

TASK_NAME="semevalenhanced"
OUTPUT_DIR=./output/${model}_enhanced_label_smoothing_task1_testac

# enhanced on task 1  hope it will work on roberta-large



CUDA_VISIBLE_DEVICES=2 python    run_roberta.py \
        --task_name $TASK_NAME \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --do_train \
        --label_smoothing \
        --do_eval \
        --data_dir $SEMEVAL_DIR_TASK1 \
        --learning_rate 5e-6 \
        --num_train_epochs 15 \
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