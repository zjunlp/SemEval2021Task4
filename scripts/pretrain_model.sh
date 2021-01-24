model="roberta-large"
MODEL_NAME_OR_PATH="/home/xx/pretrained_model/"${model}
TRAIN_FILE="./dataset/cnn-daily/train.txt"
VAL_FILE="./dataset/cnn-daily/val.txt"
OUTPUT_DIR="./output/pretrain_test3"

# -m torch.distributed.launch --nproc_per_node=1  --nnodes=1
CUDA_VISIBLE_DEVICES=0 python    \
 ./pretrain/pretrain.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_type roberta \
    --train_data_file $TRAIN_FILE \
    --eval_data_file $VAL_FILE \
    --do_train \
    --do_eval \
    --num_train_epochs 4 \
    --learning_rate  1e-6 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --output_dir $OUTPUT_DIR \
    --mlm \
    --line_by_line  \
    --block_size 256 \
    --save_steps 20000 
