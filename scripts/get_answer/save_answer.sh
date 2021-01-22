MODEL_PATH="./output/albert-xxlarge-v2_task1_128_sliding_window/checkpoint-33500"
MODEL_PATH="output/albert-xxlarge-v2_task1_128_test/checkpoint-500"

albert_1="saved_model_file/albert_task2"
albert_2="saved_model_file/albert_task2_enhanced"
roberta_1="saved_model_file/roberta_task2"
xlnet="/home/chenxn/SemEval2021/output/xlnet-large-cased_task1_accumulate16_polylr8e-6/checkpoint-2750"

CUDA_VISIBLE_DEVICES=  python save_answer.py \
    --max_seq_length 256 \
    --data_dir "./dataset/task1" \
    --model_list $xlnet \
    --overwrite_cache