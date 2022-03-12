# RNER

RNER is a Named Entity Recognition (NER) system.

It was intially used in the SemEval 2022 MULTICONER Shared Task. Since then it was used to train other NER systems for Romanian language.

For prediction out of raw text, the system uses spaCy, with the ro_core_news_lg model for Romanian language.
It can be downloaded with python -m spacy download ro_core_news_lg

The system is also able to serve pre-trained models using the "--server" command line switch.

# Acknowledgement

This code was inspired by https://github.com/mukhal/xlm-roberta-ner . In turn, that version was inspired by https://github.com/kamalkraj/BERT-NER .
On top of the original code, new layers, features and command line options were added.
