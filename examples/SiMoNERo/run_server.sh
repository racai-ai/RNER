#!/bin/sh

OUT="experiments/nou2_li_4"

python ../../src/main.py \
      --data_dir=dummy \
      --task_name=ner \
      --output_dir=$OUT \
      --max_seq_length=128   \
      --pretrained_path ../../pretrained_models/xlmr.large/ \
      --labels "O,B-DISO,I-DISO,B-CHEM,B-PROC,I-PROC,I-CHEM,B-ANAT,I-ANAT" \
      --server \
      --server_port 5110 \
      --predict_batch_size=32

