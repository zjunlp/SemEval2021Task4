
# semeval task1 for training and evaluating 
SEMEVAL_DIR="./dataset/task1"
# semeval task2
SEMEVAL_DIR_TASK2="./dataset/task2"
SEMEVAL_DIR_3="./dataset/task3_eval3"
SEMEVAL_DIR_2="./dataset/task3_eval2"
SEMEVAL_DIR_1="./dataset/task3_eval1"

task_name=($SEMEVAL_DIR $SEMEVAL_DIR_TASK2 $SEMEVAL_DIR_1 $SEMEVAL_DIR_2 $SEMEVAL_DIR_3)
# albert task1 83%  
albert_model_path="./output/albert-xxlarge-v2_512_task1"

# enhanced trained on task 1
enhanced_model_path="./output/roberta-base_enhanced"

# roberta model
roberta_model_path="./output/roberta-base_lr_9"
roberta_train_1="./output/roberta-large_128_5_task1"
ro="./best_model/roberta-task2"

# in case some models cannot run on 512
seq_len='128'


# eval on single checkpoint, add tokenizer plz

for task in ${task_name[@]}
do
CUDA_VISIBLE_DEVICES=2 python run_roberta.py \
        --task_name semeval \
        --model_name_or_path ${roberta_train_1} \
        --do_eval \
        --data_dir  $task \
        --max_seq_length ${seq_len} \
        --output_dir ${roberta_train_1} \
        --per_device_eval_batch_size=1 

echo "test on task-3 on dev"
echo ${task}
done

