MODEL_PATH="./output/albert-xxlarge-v2_512_task1"



CUDA_VISIBLE_DEVICES=0 python get_bad_cases.py \
    --model_name_or_path $MODEL_PATH \
    --max_seq_length 128 \
    --data_dir "./dataset/task2"