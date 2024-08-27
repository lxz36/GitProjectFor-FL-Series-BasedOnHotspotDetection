#!/bin/bash

unset PYTHONPATH

LOG_DIR=log/single_model
if [ ! -d $LOG_DIR ]; then
    mkdir $LOG_DIR
fi

python -u trainval_single_model.py \
    --model_path    models/model-single \
    | tee $LOG_DIR/single_asml1_iccad2012_25000steps.log
