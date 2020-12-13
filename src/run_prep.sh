#!/usr/bin/env bash

BASE_DIR=/disk1/sajad/datasets/sci/arxiv-origin/arxiv-dataset

for SET in train val
do
    python3 run_db.py -mode format_arxiv_to_lines \
                        -dataset $SET \
                        -collection arxiv \
                        -raw_path $BASE_DIR/single-files/my-format/$SET/ \
                        -save_path $BASE_DIR/jsons/  \
                        -n_cpus 8
done

#for SET in train val
#do
#    python3 run_db.py -mode format_to_bert_arxiv \
#                        -dataset $SET \
#                        -collection arxiv \
#                        -save_path $BASE_DIR/bert-files/arxiv-512/  \
#                        -raw_path $BASE_DIR/jsons/ \
#                        -n_cpus 9 \
#                        -log_file ../logs/preprocess.log
#done