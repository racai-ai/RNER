#!/bin/sh

mkdir -p experiments
export OUT="experiments/nou2_li_6/"

python ../SharedTasks/xlm-roberta-ner-work/main.py \
      --data_dir=./data_xlmr_3/ \
      --task_name=ner \
      --output_dir=$OUT \
      --max_seq_length=128   \
      --num_train_epochs 80  \
      --do_eval \
      --warmup_proportion=0.0 \
      --pretrained_path ../SharedTasks/xlm-roberta-ner-work/pretrained_models/xlmr.large/ \
      --learning_rate 0.00002 \
      --gradient_accumulation_steps 8 \
      --do_train \
      --eval_on test \
      --train_batch_size 128 \
      --eval_batch_size 32 \
      --dropout 0.1 \
      --labels "O,B-DISO,I-DISO,B-CHEM,B-PROC,I-PROC,I-CHEM,B-ANAT,I-ANAT"

cp train_li.sh $OUT
