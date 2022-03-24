#!bash

run_name=distilroberta-base-yelp-pair-b_256_16-acc64_3
mkdir -p ./output_sent_logs/$run_name
cuda_devs=1
model_name="distilroberta-base"
#model_name = "distilbert-base-cased"
#model_name = "albert-base-v2"

data_name="yelp-pair-b"
#data_name = "yelp-pair-rand-b" ## over businesses

seq_len=256
batch_size=16
acc_steps=64
num_epoch=3
cuda_devs="1"

mkdir -p ./output_sent_logs/$run_name

MLFLOW_EXPERIMENT_NAME=same-sentiment \
    CUDA_VISIBLE_DEVICES=$cuda_devs \
    python trainer.py \
    --do_train --do_eval --do_test \
    --model_name_or_path $model_name \
    --task_name same-b \
    --data_dir ./data/sentiment/$data_name \
    --output_dir ./output_sent/$run_name \
    --run_name {run_name} \
    --per_device_eval_batch_size $batch_size \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps $acc_steps \
    --logging_steps 5000 \
    --save_steps 10000 \
    --save_total_limit 3 \
    --num_train_epochs $num_epoch \
    --max_seq_length $seq_len \
    --evaluation_strategy epoch \
    > >(tee -a ./output_sent_logs/{run_name}/stdout.log) \
    2> >(tee -a ./output_sent_logs/{run_name}/stderr.log >&2)