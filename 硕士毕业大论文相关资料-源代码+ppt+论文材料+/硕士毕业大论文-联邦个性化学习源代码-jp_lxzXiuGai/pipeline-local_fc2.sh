#!/bin/bash

unset PYTHONPATH

LOG_DIR=log/fl_but_fc2
if [ ! -d $LOG_DIR ]; then
    mkdir $LOG_DIR
fi


python -u trainval_with_local_fc2.py \
    --n_iccad2012   5 \
    --n_asml1       5 \
    --sel_ratio     0.5 \
    --model_path    models/model-local_fc2-a5i5-sel0.5-ch17 \
    --top-k-channels 17 \
    | tee $LOG_DIR/a5i5_sel0.5_ch17.log


python -u trainval_with_local_fc2.py \
    --n_iccad2012   5 \
    --n_asml1       5 \
    --sel_ratio     0.5 \
    --model_path    models/model-local_fc2-a5i5-sel0.5-ch11 \
    --top-k-channels 11 \
    | tee $LOG_DIR/a5i5_sel0.5_ch11.log
