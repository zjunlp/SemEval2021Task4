model='xlnet-base-cased'

for((i=1;i<=4;i++))
do
for((j=1;j<=2;j++))
do
SEMEVAL_DIR_TASK2="./task2/task2_"${j}
OUTPUT_DIR=./output/${model}_task2_${j}_lr${i}e-5
# -m torch.distributed.launch --nproc_per_node=1  --nnodes=1

CUDA_VISIBLE_DEVICES=2 python \
        run_roberta.py \
        --task_name semeval \
        --model_name_or_path ${model} \
        --do_train \
        --do_eval \
        --data_dir $SEMEVAL_DIR_TASK2 \
        --learning_rate ${i}e-5 \
        --num_train_epochs 25 \
        --max_seq_length 384 \
        --output_dir OUTPUT_DIR \
        --lr_scheduler poly  \
        --save_steps 250 \
        --eval_steps 250 \
        --per_device_eval_batch_size=8 \
        --per_device_train_batch_size=4 \
        --gradient_accumulation_steps 4 \
        --overwrite_output  \
        --evaluate_during_training  \
done
done