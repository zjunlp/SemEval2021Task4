model='albert-xxlarge-v2'
MODEL_NAME_OR_PATH="/home/xx/pretrained_model/"${model}
SEMEVAL_DIR="./dataset/train_enhanced"
TASK_NAME="semevalenhanced"

# enhanced on task 1  hope it will work on roberta-large

CUDA_VISIBLE_DEVICES=2 python    run_roberta.py \
        --task_name $TASK_NAME \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --do_train \
        --do_eval \
        --data_dir $SEMEVAL_DIR \
        --learning_rate 1e-5 \
        --num_train_epochs 8 \
        --max_seq_length 128 \
        --output_dir ./output/${model}_enhanced \
        --save_steps 2000 \
        --eval_steps 500 \
        --per_device_eval_batch_size=8 \
        --per_device_train_batch_size=1 \
        --gradient_accumulation_steps 1 \
        --overwrite_output \
        --logging_dir ./output/${model}_enhanced \
        --evaluate_during_training  \
        --load_best_model_at_end 