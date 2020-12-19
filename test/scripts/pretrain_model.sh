model="roberta-large"
MODEL_NAME_OR_PATH="/home/xx/pretrained_model/"${model}
TRAIN_FILE="./dataset/pretrain_dataset/cnn-daily/train.txt"
VAL_FILE="./dataset/pretrain_dataset/cnn-daily/val.txt"
OUTPUT_DIR="./output/pretrain"

CUDA_VISIBLE_DEVICES= python ./pretrain/pretrain.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_type roberta \
    --train_data_file $TRAIN_FILE \
    --eval_data_file $VAL_FILE \
    --do_train \
    --do_eval \
    --output_dir $OUTPUT_DIR \
    --mlm \
    --line_by_line  \
    --overwrite_cache
