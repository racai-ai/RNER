#!/bin/sh

export LANG="hi"

mkdir -p experiments
export OUT="experiments/${LANG}_multi_mix_1/"

python ../xlm-roberta-ner-work/main.py \
      --data_dir=./data_xlmr/$LANG/ \
      --train_existing_model=./experiments/multi_mix_1/ \
      --task_name=ner \
      --output_dir=$OUT \
      --max_seq_length=128   \
      --num_train_epochs 80  \
      --do_eval \
      --warmup_proportion=0.0 \
      --pretrained_path ../xlm-roberta-ner-work/pretrained_models/xlmr.large/ \
      --learning_rate 0.00001 \
      --gradient_accumulation_steps 128 \
      --do_train \
      --eval_on test \
      --train_batch_size 128 \
      --eval_batch_size 4 \
      --dropout 0.1 \
      --labels "B-CW,I-CW,O,B-PER,I-PER,B-CORP,I-CORP,B-GRP,I-GRP,B-LOC,I-LOC,B-PROD,I-PROD"

cp ${LANG}_train_li.sh $OUT
