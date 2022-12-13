#!/usr/bin/env bash

############### Normal- experiments Longsumm #################


# CSP
#BASE_DIR=/disk1/sajad/datasets/sci/arxiv/
#RAW_PATH=$BASE_DIR/my-format-sample/
#SAVE_JSON=$BASE_DIR/my-format-sample/jsons/
#BERT_DIR=$BASE_DIR/my-format-sample/bert-files/512-section-arxiv/

# Longsumm
#BASE_DIR=/disk1/sajad/datasets/sci/longsumm/
#RAW_PATH=$BASE_DIR/my-format-splits/
#SAVE_JSON=$BASE_DIR/jsons/whole/
#BERT_DIR=$BASE_DIR/bert-files/1536-segmented-intro1024-cls/

# /disk1/sajad/datasets/sci/longsumm/bert-files/2500-segmented/

# PubMed-long
BASE_DIR=/disk1/sajad/datasets/sci/pubmedL/
RAW_PATH=$BASE_DIR/splits-with-sections-0/
SAVE_JSON=$BASE_DIR/jsons/jsons-whole-0/
#BERT_DIR=$BASE_DIR/bert-files/2500-whole-segmented-longformer-ph2/
BERT_DIR=$BASE_DIR/bert-files/2500-whole-segmented/

# arxiv-long
#BASE_DIR=/disk1/sajad/datasets/sci/arxivL/
#RAW_PATH=$BASE_DIR/splits-with-sections-introConc/
#SAVE_JSON=$BASE_DIR/jsons/whole-introConc/
#BERT_DIR=$BASE_DIR/bert-files/intro2048-segmented-15-introConc/

# csabs
#BASE_DIR=/disk1/sajad/datasets/sci/csabs/
#RAW_PATH=$BASE_DIR/my-format-splits/
#SAVE_JSON=$BASE_DIR/json/
#BERT_DIR=$BASE_DIR/bert-files/5l-csabs/

## main-arxiv
#BASE_DIR=/disk1/sajad/datasets/sci/arxiv-dataset/
#RAW_PATH=$BASE_DIR/single_files/my-format/
#SAVE_JSON=$BASE_DIR/jsons/whole/
#BERT_DIR=$BASE_DIR/bert-files/2500-whole-segmented-longformer/

# medical
#BASE_DIR=/disk1/sajad/datasets/medical/cxr/
#RAW_PATH=$BASE_DIR/splits/
#SAVE_JSON=$BASE_DIR/jsons/whole/
#BERT_DIR=$BASE_DIR/bert-files/CXR-3L/


echo "-----------------"
echo "Outputting Bart Files..."
echo "-----------------"

SAVE_JSON=$BASE_DIR/bart/whole/

for SET in test val train
do
    python3 preprocess.py -mode format_to_lines_bart \
                        -save_path $SAVE_JSON  \
                        -n_cpus 24 \
                        -log_file ../logs/preprocess.log \
                        -raw_path $RAW_PATH/$SET/ \
                        -dataset $SET \
                        -sent_numbers_path /disk1/sajad/save_lists/pubmedL-$SET-BertSumIntroGuided-top-sents.p
done



#echo "Starting to write aggregated json files..."
#echo "-----------------"
#for SET in test train val
#do
#    python3 preprocess.py -mode format_to_lines \
#                        -save_path $SAVE_JSON  \
#                        -n_cpus 24 \
#                        -keep_sect_num \
#                        -shard_size 1999 \
#                        -log_file ../logs/preprocess.log \
#                        -raw_path $RAW_PATH/$SET/ \
#                        -dataset $SET
#done

#echo "-----------------"
#echo "Now starting to write torch files..."
#echo "-----------------"
#
#for SET in train test val
#do
#    python3 preprocess.py -mode format_to_bert \
#                        -bart \
#                        -model_name scibert \
#                        -dataset $SET \
#                        -raw_path $SAVE_JSON/ \
#                        -save_path $BERT_DIR/ \
#                        -n_cpus 24 \
#                        -log_file ../logs/preprocess.log
##                        -lower \
##                        -sent_numbers_file save_lists/lsum-$SET-longformer-multi50-aftersdu-top-sents.p
#
#done
