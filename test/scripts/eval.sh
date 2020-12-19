# semeval task1 for training and evaluating 
# the result will writen in the output_dir/eval_result.txt with task name and acc


SEMEVAL_DIR_TASK1="./dataset/task1"
SEMEVAL_DIR_TASK2="./dataset/task2"
SEMEVAL_DIR_3="./dataset/task3_eval3"
SEMEVAL_DIR_2="./dataset/task3_eval2"
SEMEVAL_DIR_1="./dataset/task3_eval1"

task_name=($SEMEVAL_DIR_TASK1 $SEMEVAL_DIR_TASK2 $SEMEVAL_DIR_1 $SEMEVAL_DIR_2 $SEMEVAL_DIR_3)

albert_task2_enhanced="./saved_model_file/albert_task2_enhanced"
roberta_task1="./output/roberta-large_128_train_1_sliding_window"

# in case some models cannot run on 512
seq_len='128'
# eval on single checkpoint, add tokenizer plz
# model_name_or_path for the tokenizer
# output_dir for the pytorch.bin saved model path
roberta="/home/xx/pretrained_model/roberta-large"
albert="/home/xx/pretrained_model/albert-xxlarge-v2"
for task in ${task_name[@]}
do
CUDA_VISIBLE_DEVICES=1 python run_roberta.py \
        --sliding_window \
        --task_name semeval \
        --model_name_or_path  ${roberta} \
        --eval_all_checkpoints \
        --do_eval \
        --data_dir  $task \
        --max_seq_length ${seq_len} \
        --output_dir ${roberta_task1} \
        --per_device_eval_batch_size=1 


echo "test on task-3 on dev"
echo ${task}
done

