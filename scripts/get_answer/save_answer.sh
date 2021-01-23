<<"COMMENT"
需要修改的参数有
MAX_SEQ_LENGTH 输入的token数
MODEL_NAME_OR_PATH 模型路径需要有tokenizer
ACC 记录模型验证集分数，在融合的时候用来设置权值
OUTPUT_DIR 存储result.pkl 的路径，最好使用模型的名字

COMMENT








MODEL_PATH="./output/albert-xxlarge-v2_task1_128_sliding_window/checkpoint-33500"
MODEL_PATH="output/albert-xxlarge-v2_task1_128_test/checkpoint-500"


MAX_SEQ_LENGTH=512
MODEL_NAME_OR_PATH=""
ACC=80.1
OUTPUT_DIR="./answer_file/roberta-large"

# albert_1="saved_model_file/albert_task2"
# albert_2="saved_model_file/albert_task2_enhanced"
# roberta_1="saved_model_file/roberta_task2"
# xlnet="/home/chenxn/SemEval2021/output/xlnet-large-cased_task1_accumulate16_polylr8e-6/checkpoint-2750"

CUDA_VISIBLE_DEVICES=  python save_answer.py \
    --max_seq_length $MAX_SEQ_LENGTH \
    --data_dir "./dataset/task1" \
    --model_list $xlnet \
    --overwrite_cache