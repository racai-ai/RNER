#!/bin/sh

#export CUDA_VISIBLE_DEVICES=""

export LANG="multi"

mkdir -p experiments
export OUT="experiments/${LANG}_li_4/"

python ../xlm-roberta-multi/main.py \
      --data_dir=./data_xlmr/$LANG \
      --task_name=ner_multi \
      --output_dir=$OUT \
      --max_seq_length=128   \
      --pretrained_path ../xlm-roberta-ner-work/pretrained_models/xlmr.large/ \
      --do_predict \
      --predict_on test \
      --predict_batch_size 64 \
      --predict_format ann_only \
      --predict_filename "$LANG.pred.conll" \
      --labels "bn,de,en,es,fa,hi,ko,nl,ru,tr,zh" \
      --ner_labels  "B-CW,I-CW,O,B-PER,I-PER,B-CORP,I-CORP,B-GRP,I-GRP,B-LOC,I-LOC,B-PROD,I-PROD"

