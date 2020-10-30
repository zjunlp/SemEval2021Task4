# bert-base-uncased
# roberta-base
# xlnet-base-cased
# longformer-base-4096

model='roberta-base'



MODEL_NAME_OR_PATH="/home/xx/pretrained_model/longformer-base-4096"
MODEL_NAME_OR_PATH="/home/xx/pretrained_model/bert-base-uncased"
MODEL_NAME_OR_PATH="/home/xx/pretrained_model/roberta-base"

MODEL_NAME_OR_PATH="/home/xx/pretrained_model/"${model}


SEMEVAL_DIR="./dataset/training_data"

model_path='./model_saved/44'
model_path='./output/roberta-base_enhanced/checkpoint-14000'
# model_path='./output/roberta-base_5cls_2'

CUDA_VISIBLE_DEVICES=1  python run_roberta.py \
        --task_name semevalenhanced \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --do_eval \
        --eval_all_checkpoints \
        --data_dir $SEMEVAL_DIR \
        --learning_rate 4e-2 \
        --num_train_epochs 10 \
        --max_seq_length 512 \
        --output_dir ${model_path} \
        --save_steps 1000 \
        --per_device_eval_batch_size=8 \
        --per_device_train_batch_size=1 \
        --gradient_accumulation_steps 1 








