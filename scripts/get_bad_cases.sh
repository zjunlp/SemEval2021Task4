MODEL_PATH="./saved_model_file/roberta_task2"



CUDA_VISIBLE_DEVICES=1 python get_bad_cases.py \
    --model_name_or_path $MODEL_PATH \
    --max_seq_length 128 \
    --data_dir "./dataset/task2"