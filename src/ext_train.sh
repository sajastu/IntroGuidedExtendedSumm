#!/usr/bin/env bash

################################################################################
##### CHECKPOINTS –to train from #######
################################################################################
#CHECKPOINT=/disk1/sajad/sci-trained-models/presum/arxiv-first-phase/model_step_30000.pt


################################################################################
##### Global Params #######
################################################################################
DATASET=pubmedL
N_SENTS=25
VAL_INTERVAL=4000

METHOD=baseline
export CUDA_VISIBLE_DEVICES=0

MODEL_NAME=longformer
MAX_POS=2500
BSZ=1


################################################################################
##### Data #######
################################################################################

DATA_PATH=/disk1/sajad/datasets/sci/$DATASET/bert-files/2500-segmented-seqLabelled-$N_SENTS/

################################################################################
##### MODEL #######
################################################################################

MODEL_PATH=/disk1/sajad/sci-trained-models/presum/$DATASET-2500-segmented-$METHOD-classi-v1/

################################################################################
##### TRAINING SCRIPT #######
################################################################################


LOG_DIR=../logs/$(echo $MODEL_PATH | cut -d \/ -f 6).log
mkdir -p ../results/$(echo $MODEL_PATH | cut -d \/ -f 6)
RESULT_PATH_TEST=../results/$(echo $MODEL_PATH | cut -d \/ -f 6)/

if [[ "$METHOD" == *"multi"* ]]; then

    python3 train.py -task ext \
                    -mode train \
                    -model_name $MODEL_NAME \
                    -val_pred_len $N_SENTS \
                    -bert_data_path $DATA_PATH \
                    -ext_dropout 0.1 \
                    -model_path $MODEL_PATH \
                    -lr 2e-3 \
                    -visible_gpus $CUDA_VISIBLE_DEVICES \
                    -report_every 50 \
                    -log_file $LOG_DIR \
                    -val_interval $VAL_INTERVAL \
                    -save_checkpoint_steps 200000 \
                    -batch_size $BSZ \
                    -test_batch_size 5000 \
                    -max_length 600 \
                    -train_steps 300000 \
                    -alpha 0.95 \
                    -use_interval true \
                    -warmup_steps 10000 \
                    -max_pos $MAX_POS \
                    -result_path_test $RESULT_PATH_TEST \
                    -accum_count 2 \
                    -saved_list_name /disk1/sajad/save_lists/pubmedL-val-longformer-multi50.p \
                    -section_prediction \
                    -alpha_mtl 0.50
else
    python3 train.py -task ext \
                    -mode train \
                    -model_name longformer \
                    -val_pred_len $N_SENTS \
                    -bert_data_path $DATA_PATH \
                    -ext_dropout 0.1 \
                    -model_path $MODEL_PATH \
                    -lr 2e-3 \
                    -visible_gpus $CUDA_VISIBLE_DEVICES \
                    -report_every 50 \
                    -log_file $LOG_DIR \
                    -val_interval $VAL_INTERVAL \
                    -save_checkpoint_steps 300000 \
                    -batch_size $BSZ \
                    -test_batch_size 5000 \
                    -max_length 600 \
                    -train_steps 200000 \
                    -saved_list_name /disk1/sajad/save_lists/pubmedL-val-longformer-baseline.p \
                    -alpha 0.95 \
                    -use_interval true \
                    -warmup_steps 10000 \
                    -max_pos $MAX_POS \
                    -result_path_test $RESULT_PATH_TEST \
                    -accum_count 2

fi