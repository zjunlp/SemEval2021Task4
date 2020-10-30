model='roberta-base'
MODEL_NAME_OR_PATH="/home/xx/pretrained_model/"${model}
SEMEVAL_DIR="./dataset/training_data"

# about 1h for test a learning_rate

for((i=5;i<20;i+=1));
do
echo ==========training with learning rate ${i}e-6===========;

CUDA_VISIBLE_DEVICES=1 python \
        run_roberta.py \
        --task_name semeval \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --do_train \
        --do_eval \
        --data_dir $SEMEVAL_DIR \
        --learning_rate ${i}e-6 \
        --num_train_epochs 10 \
        --max_seq_length 512 \
        --output_dir ./output/${model}_lr_${i} \
        --save_steps 2000 \
        --per_device_eval_batch_size=8 \
        --per_device_train_batch_size=1 \
        --gradient_accumulation_steps 1 \
        --overwrite_output  \

done









# baseline
# CUDA_VISIBLE_DEVICES=2 python ./Baselines/Run_GAReader.py