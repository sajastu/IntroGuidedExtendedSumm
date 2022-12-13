#!/usr/bin/env bash



#########################
######### Data #########
#########################

BERT_DIR=/disk1/sajad/datasets/sci/arxivL//bert-files/2048-segmented-intro1536-15-introConc/
#BERT_DIR=/disk1/sajad/datasets/sci/pubmedL/bert-files/2048-segmented-intro2048-20-introConc/

#########################
######### MODELS#########
#########################

#number of intro_samples should differ for each dataset!!!!!

MODEL_PATH=/disk1/sajad/sci-trained-models/presum/ARXIVL-2048-introG1536-IntroConc/ #final model
#MODEL_PATH=/disk1/sajad/sci-trained-models/presum/pubmedL-2048-segmented-introG2048Baseline-IntroConc/ #final model

CHECKPOINT=$MODEL_PATH/BEST_model_s70000_0.4937_0.2046_0.2231.pt #arxivL-ch
#CHECKPOINT=$MODEL_PATH/BEST_model_s190000_0.4952_0.2155_0.2392.pt #pubmedL-ch

export CUDA_VISIBLE_DEVICES=1

MAX_POS=2048
MAX_POS_INTRO=1536

RG_CELL=H90:J90

mkdir -p $MODEL_PATH/results/

for ST in val test train
do
#    RESULT_PATH=../results/$(echo $MODEL_PATH | cut -d \/ -f 6)/abs-set/$ST.official
#    RESULT_PATH=/home/sajad/datasets/longsum/submission_files/
    mkdir -p $MODEL_PATH/results/$ST
    RESULT_PATH=$MODEL_PATH/results/$ST
    python3 train.py -task ext \
                    -mode test \
                    -test_batch_size 32000 \
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
                    -min_length 600 \
                    -finetune_bert False \
                    -result_path $RESULT_PATH \
                    -test_from $CHECKPOINT \
                    -model_name longformer \
                    -val_pred_len 20 \
                    -gd_cells_rg $RG_CELL \
                    -gd_cell_step R90 \
                    -saved_list_name /disk1/sajad/save_lists/pubmedL-$ST-BertSumIntroGuided.p \
                    -section_prediction \
                    -pick_top

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


