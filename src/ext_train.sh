#!/usr/bin/env bash

################################################################################
##### CHECKPOINTS –to train from #######
################################################################################
#CHECKPOINT=/disk1/sajad/sci-trained-models/presum/arxiv-first-phase/model_step_30000.pt


################################################################################
##### Global Params #######
################################################################################
export DATASET=arxivL
export N_SENTS=15
export TRAINING_STEPS=100000
export VAL_INTERVAL=5000

METHOD=introG1536-IntroConc
export CUDA_VISIBLE_DEVICES=0

MODEL_NAME=longformer
MAX_POS=2048
MAX_POS_INTRO=1536


ROW_NUMBER=100
export GD_CELLS_RG_VAL=D$ROW_NUMBER:F$ROW_NUMBER
export GD_CELLS_RG_TEST=H$ROW_NUMBER:J$ROW_NUMBER

export GD_CELLS_RECALL_VAL=G$ROW_NUMBER
export GD_CELLS_RECALL_TEST=K$ROW_NUMBER

export GD_CELLS_STEP=Q$ROW_NUMBER

BSZ=4000

################################################################################
##### Data #######
################################################################################

DATA_PATH=/disk1/sajad/datasets/sci/$DATASET/bert-files/$MAX_POS-segmented-intro$MAX_POS_INTRO-$N_SENTS-introConc/
################################################################################
##### MODEL #######
################################################################################
MODEL_DB=`echo $DATASET | tr 'a-z' 'A-Z'`
MODEL_PATH=/disk1/sajad/sci-trained-models/presum/$MODEL_DB-$MAX_POS-$METHOD/

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
                    -intro_cls \
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
                    -train_steps $TRAINING_STEPS \
                    -saved_list_name /disk1/sajad/save_lists/pubmedL-val-longformer-baseline.p \
                    -alpha 0.95 \
                    -use_interval true \
                    -warmup_steps 10000 \
                    -max_pos $MAX_POS \
                    -gd_cells_rg $GD_CELLS_RG_VAL \
                    -gd_cell_step $GD_CELLS_STEP \
                    -max_pos_intro $MAX_POS_INTRO\
                    -result_path_test $RESULT_PATH_TEST \
                    -accum_count 2 \
#                    -train_from /disk1/sajad/sci-trained-models/presum/arxivL-2500-segmented-introG1536/model_step_4000.pt \

fi