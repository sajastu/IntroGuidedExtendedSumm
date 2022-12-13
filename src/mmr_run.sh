#!/usr/bin/env bash




### see effects of co2
#for C2 in .2 .3 .4
#do
#    python pick_mmr.py -co1 .9 \
#                        -co2 $C2 \
#                        -co3 0.1 \
#                        -cos
#done


##########################
## see effects of co1 and co2  ##
#########################
#C1=.9
#C3=0.05
#C2=0.05
#echo "For $C1, $C2, $C3:"
#python pick_mmr.py -co1 $C1 \
#                        -co2 $C2 \
#                        -co3 $C3
#
#C1=.8
#C3=0.1
#C2=0.1
#echo "For $C1, $C2, $C3:"
#python pick_mmr.py -co1 $C1 \
#                        -co2 $C2 \
#                        -co3 $C3
#
#C1=.8
#C3=0.05
#C2=0.15
#echo "For $C1, $C2, $C3:"
#python pick_mmr.py -co1 $C1 \
#                        -co2 $C2 \
#                        -co3 $C3
#
#
#C1=.8
#C3=0.15
#C2=0.05
#echo "For $C1, $C2, $C3:"
#python pick_mmr.py -co1 $C1 \
#                        -co2 $C2 \
#                        -co3 $C3

##########################
## see effects of co2  ##
#########################


#C1=.95
#C3=0
#C2=.05
#echo "For $C1, $C2, $C3:"
#python pick_mmr.py -co1 $C1 \
#                        -co2 $C2 \
#                        -co3 $C3
#
#C1=.9
#C3=0
#C2=.1
#echo "For $C1, $C2, $C3:"
#python pick_mmr.py -co1 $C1 \
#                        -co2 $C2 \
#                        -co3 $C3
#
#
#C1=.8
#C3=0
#C2=.2
#echo "For $C1, $C2, $C3:"
#python pick_mmr.py -co1 $C1 \
#                        -co2 $C2 \
#                        -co3 $C3

##########################
## see effects of co3  ##
#########################
#

#C1=.9
#C2=0
#C3=.1
#echo "For $C1, $C2, $C3:"
#python pick_mmr.py -co1 $C1 \
#                        -co2 $C2 \
#                        -co3 $C3

#C1=.95
#C2=0
#C3=.05
#echo "For $C1, $C2, $C3:"
#python pick_mmr.py -co1 $C1 \
#                        -co2 $C2 \
#                        -co3 $C3
#
#


#for SET in test
#do
#
#    PRED_LEN=30
#    METHOD=_base
#    SAVED_LIST=save_lists/LSUM-$SET-longformer-multi.p
#
#    C1=.8
#    C2=0
#    C3=0.2
#    echo "For $C1, $C2, $C3:"
#    python pick_mmr.py -co1 $C1 \
#                            -co2 $C2 \
#                            -co3 $C3 \
#                            -set $SET \
#                            -method $METHOD \
#                            -pred_len $PRED_LEN \
#                            -cos \
#                            -saved_list $SAVED_LIST
#
##    C1=.8
##    C2=0
##    C3=0.2
##    echo "For $C1, $C2, $C3:"
##    python pick_mmr.py -co1 $C1 \
##                            -co2 $C2 \
##                            -co3 $C3 \
##                            -set $SET \
##                            -method $METHOD \
##                            -pred_len $PRED_LEN \
##                            -cos \
##                            -saved_list $SAVED_LIST
##
##    C1=.7
##    C2=0
##    C3=.3
##    echo "For $C1, $C2, $C3:"
##    python pick_mmr.py -co1 $C1 \
##                            -co2 $C2 \
##                            -co3 $C3 \
##                            -set $SET \
##                            -method $METHOD \
##                            -pred_len $PRED_LEN \
##                            -cos \
##                            -saved_list $SAVED_LIST \
##                            -end
#
#done



for SET in test
do
    for MT in _base
    do

        PRED_LEN=15
        METHOD=_base
#        save_list_arxiv_long_test_sectioned_bertsum.p
#        SAVED_LIST=save_lists/pubmedL-$SET-longformer-$MT.p
        SAVED_LIST=/disk1/sajad/save_lists/arxivL-$SET-BertSumExt.p
#        SAVED_LIST=/disk1/sajad/save_lists/arxivL-$SET-BertSumIntroGuided.p
#        SAVED_LIST=save_lists/LSUM-$SET-sectioned-$MT.p
        C1=.8
        C2=0
        C3=0.2
        python pick_mmr.py -co1 $C1 \
                                -co2 $C2 \
                                -co3 $C3 \
                                -set $SET \
                                -method $METHOD \
                                -pred_len $PRED_LEN \
                                -saved_list $SAVED_LIST


        SAVED_LIST=/disk1/sajad/save_lists/arxivL-$SET-BertSumIntroGuided.p
        python pick_mmr.py -co1 $C1 \
                                -co2 $C2 \
                                -co3 $C3 \
                                -set $SET \
                                -method $METHOD \
                                -pred_len $PRED_LEN \
                                -saved_list $SAVED_LIST




    #                            -cos \

    #    PRED_LEN=25
    #    METHOD=_mmr
    #    SAVED_LIST=save_lists/pumbedL-$SET-longformer-multi-lowR.p
    #
    #    C1=.9
    #    C2=0
    #    C3=0.1
    #    echo "For $C1, $C2, $C3:"
    #    python pick_mmr.py -co1 $C1 \
    #                            -co2 $C2 \
    #                            -co3 $C3 \
    #                            -set $SET \
    #                            -method $METHOD \
    #                            -pred_len $PRED_LEN \
    #                            -saved_list $SAVED_LIST
    ##                            -cos \
    #
    #    C1=.8
    #    C2=0
    #    C3=0.2
    #    echo "For $C1, $C2, $C3:"
    #    python pick_mmr.py -co1 $C1 \
    #                            -co2 $C2 \
    #                            -co3 $C3 \
    #                            -set $SET \
    #                            -method $METHOD \
    #                            -pred_len $PRED_LEN \
    #                            -saved_list $SAVED_LIST
    ##                            -cos \
    #
    #    C1=.7
    #    C2=0
    #    C3=.3
    #    echo "For $C1, $C2, $C3:"
    #    python pick_mmr.py -co1 $C1 \
    #                            -co2 $C2 \
    #                            -co3 $C3 \
    #                            -set $SET \
    #                            -method $METHOD \
    #                            -pred_len $PRED_LEN \
    #                            -saved_list $SAVED_LIST \
    #                            -end
    ##                            -cos \
    done
done
