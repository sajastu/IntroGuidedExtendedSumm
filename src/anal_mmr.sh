#!/usr/bin/env bash

C1=.8
C2=0
C3=.2
echo "For $C1, $C2, $C3:"
python pick_mmr_anal.py -co1 $C1 \
                        -co2 $C2 \
                        -co3 $C3
