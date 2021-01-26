<<"COMMENT"
需要修改的参数有
MAX_SEQ_LENGTH 输入的token数
MODEL_NAME_OR_PATH 模型路径需要有tokenizer
ACC 记录模型验证集分数，在融合的时候用来设置权值 ，该值自动生成，不用固安
OUTPUT_DIR 存储result.pkl 的路径，最好使用模型的名字



data_dir/train.jsonl, dev.jsonl test.jsonl
这个test git pull 一下就好了

COMMENT

#84.4
MAX_SEQ_LENGTH=256
MODEL_NAME_OR_PATH="output/albert-xxlarge-v2_pretrained_model_128_label_smoothing"
MODEL_NAME_OR_PATH="output/roberta-large_256_test_lr3"
OUTPUT_DIR="./answer_file/roberta_smooth_label_task2__test"

# albert_1="saved_model_file/albert_task2"
# albert_2="saved_model_file/albert_task2_enhanced"
# roberta_1="saved_model_file/roberta_task2"
# xlnet="/home/chenxn/SemEval2021/output/xlnet-large-cased_task1_accumulate16_polylr8e-6/checkpoint-2750"

CUDA_VISIBLE_DEVICES=3  python save_answer.py \
    --task_name semeval \
    --max_seq_length $MAX_SEQ_LENGTH \
    --data_dir "./dataset/task2" \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR  \
    --overwrite_cache