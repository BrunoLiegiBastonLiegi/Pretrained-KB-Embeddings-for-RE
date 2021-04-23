#!/bin/bash

for i in `seq 0 9`
do
    echo '\n> Running on fold: '$i
    python ADE.py DRUG-AE_BIOES_10-fold.pkl\
	   --fold $i\
	   #--load_model tmp.pth
done
