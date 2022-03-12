#!/bin/sh

export LANG="tr"

rm experiments/${LANG}_multi_mix_1/${LANG}_dev.zip
zip -j experiments/${LANG}_multi_mix_1/${LANG}_dev.zip experiments/${LANG}_multi_mix_1/${LANG}.pred.conll
