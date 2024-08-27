#!/bin/bash

unset PYTHONPATH

LOG_DIR=log/fl_but_fc12_ht.v2
if [ ! -d $LOG_DIR ]; then
    mkdir $LOG_DIR
fi


python -u trainval_with_local_fc12.py \
    --n_iccad2012   2 \
    --n_asml1       2 \
    --sel_ratio     1.0 \
    --fc1-size  250 250 50 50 \
    --model_path    models/model-local_fc12_ht.v2-a2i2-sel1.0-ch32 \
    | tee $LOG_DIR/a2i2_sel1.0_ch32.log

python -u trainval_with_local_fc12.py \
    --n_iccad2012   5 \
    --n_asml1       5 \
    --sel_ratio     1.0 \
    --fc1-size  250 250 250 250 250 50 50 50 50 50 \
    --model_path    models/model-local_fc12_ht.v2-a5i5-sel1.0-ch32 \
    | tee $LOG_DIR/a5i5_sel1.0_ch32.log

