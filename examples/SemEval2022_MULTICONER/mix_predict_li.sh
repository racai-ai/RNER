#!/bin/sh

#export CUDA_VISIBLE_DEVICES=""

export LANG="mix"

mkdir -p experiments
export OUT="experiments/${LANG}_multi_mix_1/"

python ../xlm-roberta-ner-work/main.py \
      --data_dir=./data_xlmr/$LANG \
      --task_name=ner \
      --output_dir=$OUT \
      --max_seq_length=128   \
      --pretrained_path ../xlm-roberta-ner-work/pretrained_models/xlmr.large/ \
      --do_predict \
      --predict_on test \
      --predict_batch_size 64 \
      --predict_format ann_only \
      --predict_filename "$LANG.pred.conll" \
      --labels  "B-CW,I-CW,O,B-PER,I-PER,B-CORP,I-CORP,B-GRP,I-GRP,B-LOC,I-LOC,B-PROD,I-PROD"

