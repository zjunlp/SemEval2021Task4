MAX_SEQ_LENGTH=256
MODEL_NAME_OR_PATH="/home/chenxn/SemEval2021/output/xlnet-large-cased_task2_accumulate16_polylr1.5/checkpoint-4625"

CUDA_VISIBLE_DEVICES=1 python get_acc.py \
    --task_name semeval \
    --max_seq_length $MAX_SEQ_LENGTH \
    --data_dir "./dataset/task2" \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --overwrite_cache
