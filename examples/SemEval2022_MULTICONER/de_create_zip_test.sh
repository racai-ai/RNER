#!/bin/sh

export LANG="de"

rm experiments/${LANG}_multi_mix_1/${LANG}_test.zip
zip -j experiments/${LANG}_multi_mix_1/${LANG}_test.zip experiments/${LANG}_multi_mix_1/${LANG}.pred.conll
