MODEL_PATH="./output/albert-xxlarge-v2_task1_128_sliding_window/checkpoint-33500"
MODEL_PATH="output/albert-xxlarge-v2_task1_128_test/checkpoint-500"

albert_1="saved_model_file/albert_task2"
albert_2="saved_model_file/albert_task2_enhanced"
roberta_1="saved_model_file/roberta_task2"


CUDA_VISIBLE_DEVICES=0 python save_answer.py \
    --max_seq_length 128 \
    --data_dir "./dataset/task1" \
    --model_list $albert_1 $albert_2 $roberta_1 \
    --overwrite_cache