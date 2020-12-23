model='roberta-base'



MODEL_NAME_OR_PATH="/home/xx/pretrained_model/longformer-base-4096"
MODEL_NAME_OR_PATH="/home/xx/pretrained_model/bert-base-uncased"
MODEL_NAME_OR_PATH="/home/xx/pretrained_model/roberta-base"

MODEL_NAME_OR_PATH="/home/xx/pretrained_model/"${model}


SEMEVAL_DIR="./dataset/training_data"

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
                --nproc_per_node=1 \
                --nnodes=1  \
        run_roberta.py \
        --task_name semeval \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --do_train \
        --do_eval \
        --eval_all_checkpoints \
        --data_dir $SEMEVAL_DIR \
        --learning_rate 1e-6 \
        --num_train_epochs 10 \
        --max_seq_length 512 \
        --output_dir ./output/${model}_5cls_2 \
        --modified \
        --save_steps 1000 \
        --per_device_eval_batch_size=8 \
        --per_device_train_batch_size=1 \
        --gradient_accumulation_steps 1 \
        --overwrite_output










# baseline
# CUDA_VISIBLE_DEVICES=2 python ./Baselines/Run_GAReader.py