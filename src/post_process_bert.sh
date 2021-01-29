#!/usr/bin/env bash

N_SENTS=15
DATASET=arxivL
#PT_DIRS=/disk1/sajad/datasets/sci/$DATASET/bert-files/2500-whole-segmented
#PT_DIRS=/disk1/sajad/datasets/sci/pubmedL/bert-files/2048-segmented-intro2048-25-0-8
PT_DIRS=/disk1/sajad/datasets/sci/arxivL//bert-files/2048-segmented-intro1536-15

#PT_DIRS=/disk1/sajad/datasets/sci/pubmedL-long/bert-files/2500-whole-segmented-seqLabelled
PT_DIRS_DEST=$PT_DIRS-$N_SENTS

#PT_DIRS_DEST=$PT_DIRS-updated15
#
##PT_DIRS_DEST=/disk1/sajad/datasets/sci/arxiv-long/v1/bert-files/512-segmented-sectioned-scibert-15-seqLabelled
##
#for SET in train
#do
#
#    echo "Modifying labels... for $SET"
#    python3 modify_sent_labels_bertfiles.py -pt_dirs_src $PT_DIRS \
#            -write_to $PT_DIRS_DEST \
#            -set $SET \
#            -n_sents $N_SENTS \
#            -greedy True
#
#done
#

#exit 0
PT_DIRS_DEST=/disk1/sajad/datasets/sci/arxivL//bert-files/2048-segmented-intro1536-15-introConc/

for SET in val test
do
    echo "Calculating oracle...for $SET"
    python3 calculate_oracle_from_bertfiles.py -pt_dirs_src $PT_DIRS_DEST \
            -set $SET
done

############################################################
#DATASET=arxivL
#PT_DIRS_DEST=$PT_DIRS
#SINGLE_FILES_PATH=/disk1/sajad/datasets/sci/longsumm/my-format-splits/
#for SET in train
#do
##    rm /home/sajad/introGSumm/src/$DATASET.$SET.jsonl
##    python3 convert_bert_data_to_sequential.py -read_from $PT_DIRS_DEST \
##            -dataset $DATASET \
##            -set $SET \
##            -single_json_base $SINGLE_FILES_PATH
###            -selective papers_labels_nf_$SET.txt \
##
##
##    export CUDA_VISIBLE_DEVICES=0
##    cd /home/sajad/packages/sequential_sentence_classification
###    sh scripts/predict.sh $DATASET $SET
##    sh scripts/predict.sh $DATASET /home/sajad/introGSumm/src/$DATASET.$SET.jsonl
##
##    cd /home/sajad/introGSumm/src
#
#    PREDICTED_LABELS=/home/sajad/packages/sequential_sentence_classification/$DATASET.long20k.$SET.json
#    python3 change_labels_sequential.py -read_from $PT_DIRS_DEST \
#            -write_to $PT_DIRS_DEST \
#            -predicted_labels $PREDICTED_LABELS \
#            -set $SET
###
##
#done
#######


# chunk bert files
#python3 spliting_bertfiles.py -pt_dirs_src $PT_DIRS_DEST
