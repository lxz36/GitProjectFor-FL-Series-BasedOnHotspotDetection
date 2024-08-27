#!/bin/bash

unset PYTHONPATH

LOG_DIR=log/no_server
if [ ! -d $LOG_DIR ]; then
    mkdir $LOG_DIR
fi

python -u trainval_no_server.py \
    --n_iccad2012   1 \
    --n_asml1       1 \
    --model_path    models/model-no_server-a1i1 \
    | tee $LOG_DIR/a1i1-50r-800spr.log

python -u trainval_no_server.py \
    --n_iccad2012   2 \
    --n_asml1       2 \
    --model_path    models/model-no_server-a2i2 \
    | tee $LOG_DIR/a2i2-50r-800spr.log

python -u trainval_no_server.py \
    --n_iccad2012   5 \
    --n_asml1       5 \
    --model_path    models/model-no_server-a5i5 \
    | tee $LOG_DIR/a5i5-50r-800spr.log

