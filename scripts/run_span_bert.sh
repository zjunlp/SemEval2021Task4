model="spanbert-large-cased"
MODEL_NAME_OR_PATH="/home/xx/pretrained_model/"${model}


SEMEVAL_DIR="./dataset/task1"
SEMEVAL_DIR_TASK2="./dataset/task2"

# -m torch.distributed.launch --nproc_per_node=1  --nnodes=1

CUDA_VISIBLE_DEVICES=1 python \
        run_roberta.py \
        --task_name semeval \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --sliding_window \
        --do_train \
        --do_eval \
        --sliding_window \
        --eval_all_checkpoints \
        --data_dir $SEMEVAL_DIR \
        --learning_rate 1e-5 \
        --num_train_epochs 4 \
        --max_seq_length 128 \
        --output_dir ./output/${model}_512_task1_test \
        --save_steps 500 \
        --per_device_eval_batch_size=8 \
        --per_device_train_batch_size=2 \
        --gradient_accumulation_steps 4 \
        --load_best_model_at_end  \
        --evaluate_during_training   \
        --overwrite_output_dir  \
        --eval_steps 500 











# baseline
# CUDA_VISIBLE_DEVICES=2 python ./Baselines/Run_GAReader.py