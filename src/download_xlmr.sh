#!/bin/sh


echo "This script will download XLM-ROBERTA models (base and large) to the folder pretrained_models"
echo "Press <ENTER> to continue, CTRL+C to cancel"

read a

mkdir -p pretrained_models 

PARAM_SET="base"
wget -P pretrained_models https://dl.fbaipublicfiles.com/fairseq/models/xlmr.$PARAM_SET.tar.gz
tar xzvf pretrained_models/xlmr.$PARAM_SET.tar.gz  --directory pretrained_models/
#rm -r pretrained_models/xlmr.$PARAM_SET.tar.gz

PARAM_SET="large"
wget -P pretrained_models https://dl.fbaipublicfiles.com/fairseq/models/xlmr.$PARAM_SET.tar.gz
tar xzvf pretrained_models/xlmr.$PARAM_SET.tar.gz  --directory pretrained_models/
#rm -r pretrained_models/xlmr.$PARAM_SET.tar.gz
