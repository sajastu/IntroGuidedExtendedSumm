#!/usr/bin/env bash



#########################
######### Data #########
#########################

#BERT_DIR=/disk1/sajad/datasets/sci/arxivL//bert-files/2048-segmented-intro1536-15-introConc-updated/
BERT_DIR=/disk1/sajad/datasets/sci/arxivL/bert-files/2048-segmented-intro1536-15-introConc-updated/


#########################
######### MODELS#########
#########################

#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/arxivL-2500-segmented-introG1536/
MODEL_PATH=/disk1/sajad/sci-trained-models/presum/pubmedL-2048-segmented-introG2048Baseline-IntroConc/
#/disk1/sajad/sci-trained-models/presum/pubmedL-2048-segmented-introG2048Baseline-IntroConc/

CHECKPOINT=$MODEL_PATH/model_step_50000.pt





export CUDA_VISIBLE_DEVICES=0

MAX_POS=2048
MAX_POS_INTRO=2048

RG_CELL=H90:J90

mkdir -p ../results/$(echo $MODEL_PATH | cut -d \/ -f 6)
for ST in test
do
    RESULT_PATH=../results/$(echo $MODEL_PATH | cut -d \/ -f 6)/$ST
#    RESULT_PATH=../results/$(echo $MODEL_PATH | cut -d \/ -f 6)/abs-set/$ST.official
#    RESULT_PATH=/home/sajad/datasets/longsum/submission_files/
    python3 train.py -task ext \
                    -mode test \
                    -test_batch_size 10000 \
                    -bert_data_path $BERT_DIR \
                    -log_file ../logs/val_ext \
                    -model_path $MODEL_PATH \
                    -sep_optim true \
                    -use_interval true \
                    -visible_gpus $CUDA_VISIBLE_DEVICES \
                    -max_pos $MAX_POS \
                    -max_pos_intro $MAX_POS_INTRO \
                    -max_length 600 \
                    -alpha 0.95 \
                    -exp_set $ST \
                    -pick_top \
                    -min_length 600 \
                    -finetune_bert False \
                    -result_path $RESULT_PATH \
                    -test_from $CHECKPOINT \
                    -model_name longformer \
                    -val_pred_len 20 \
                    -gd_cells_rg $RG_CELL \
                    -gd_cell_step R72 \
                    -saved_list_name save_lists/lsum-$ST-longformer-multi50-aftersdu.p \
                    -section_prediction \
                    -alpha_mtl 0.50

done

#for ST in test
#do
#    PRED_LEN=20
#    METHOD=_base
#    SAVED_LIST=save_lists/pubmedL-$ST-scibert-bertsum.p
#    C1=.8
#    C2=0
#    C3=0.2
#    python3 pick_mmr.py -co1 $C1 \
#                            -co2 $C2 \
#                            -co3 $C3 \
#                            -set $ST \
#                            -method $METHOD \
#                            -pred_len $PRED_LEN \
#                            -saved_list $SAVED_LIST \
#                            -end
#done
#


