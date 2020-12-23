MODEL_PATH="./output/albert-xxlarge-v2_task1_128_sliding_window/checkpoint-33500"
MODEL_PATH="output/albert-xxlarge-v2_task1_128_test/checkpoint-500"



CUDA_VISIBLE_DEVICES= python get_bad_cases.py \
    --model_name_or_path $MODEL_PATH \
    --max_seq_length 128 \
    --data_dir "./dataset/task1"