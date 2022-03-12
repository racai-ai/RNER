#!/bin/sh

export LANG="tr"

rm experiments/${LANG}_multi_mix_1/${LANG}_test.zip
zip -j experiments/${LANG}_multi_mix_1/${LANG}_test.zip experiments/${LANG}_multi_mix_1/${LANG}.pred.conll
