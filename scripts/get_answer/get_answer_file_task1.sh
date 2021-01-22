MODEL_PATH="./output/albert-xxlarge-v2_task1_128_sliding_window/checkpoint-33500"
MODEL_PATH="output/albert-xxlarge-v2_task1_128_test/checkpoint-500"

albert_1="saved_model_file/albert_task1"
albert_2="saved_model_file/albert_task1_best"
roberta_1="saved_model_file/roberta_task1_enhanced"


CUDA_VISIBLE_DEVICES=2 python get_bad_cases.py \
    --max_seq_length 128 \
    --data_dir "./dataset/task1_test" \
    --model_list $albert_1 $albert_2 $roberta_1 \
    --overwrite_cache