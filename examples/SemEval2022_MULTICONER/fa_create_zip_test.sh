#!/bin/sh

export LANG="fa"

rm experiments/${LANG}_multi_mix_1/${LANG}_test.zip
zip -j experiments/${LANG}_multi_mix_1/${LANG}_test.zip experiments/${LANG}_multi_mix_1/${LANG}.pred.conll
