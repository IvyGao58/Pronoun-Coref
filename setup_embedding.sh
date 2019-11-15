#!/bin/bash

# Download pretrained embeddings.
curl -O http://lsz-gpu-01.cs.washington.edu/resources/glove_50_300_2.txt
curl -O https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip

# python filter_embeddings.py glove.840B.300d.txt data/train.jsonlines data/dev.jsonlines data/test.jsonlines
python filter_embeddings.py chinese_emb/sgns.zhihu.word data/law/train.jsonlines data/law/dev.jsonlines data/law/test.jsonlines
