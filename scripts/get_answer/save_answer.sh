<<"COMMENT"
需要修改的参数有
MAX_SEQ_LENGTH 输入的token数
MODEL_NAME_OR_PATH 模型路径需要有tokenizer
ACC 记录模型验证集分数，在融合的时候用来设置权值 ，该值自动生成，不用固安
OUTPUT_DIR 存储result.pkl 的路径，最好使用模型的名字



data_dir/train.jsonl, dev.jsonl test.jsonl
这个test git pull 一下就好了

COMMENT


MAX_SEQ_LENGTH=128
MODEL_NAME_OR_PATH="output/roberta-large_enhanced_label_smoothing_task1"
OUTPUT_DIR="./answer_file/task1_enhanced_roberta_label_what_128"

# albert_1="saved_model_file/albert_task2"
# albert_2="saved_model_file/albert_task2_enhanced"
# roberta_1="saved_model_file/roberta_task2"
# xlnet="/home/chenxn/SemEval2021/output/xlnet-large-cased_task1_accumulate16_polylr8e-6/checkpoint-2750"

CUDA_VISIBLE_DEVICES=  python save_answer.py \
    --task_name semeval \
    --max_seq_length $MAX_SEQ_LENGTH \
    --data_dir "./dataset/task1" \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR \
    --overwrite_cache