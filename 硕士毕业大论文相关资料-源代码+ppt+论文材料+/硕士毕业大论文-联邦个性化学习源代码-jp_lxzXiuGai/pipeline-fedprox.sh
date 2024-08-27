#!/bin/bash

unset PYTHONPATH

LOG_DIR=log/fedprox
if [ ! -d $LOG_DIR ]; then
    mkdir $LOG_DIR
fi

python -u trainval_global_fedprox.py \
    --n_iccad2012   1 \
    --n_asml1       1 \
    --sel_ratio     1.0 \
    --model_path    models/model-fedprox-a1i1-sel1.0-l2fp1e-3-withl2 \
    | tee $LOG_DIR/fedprox-a1i1-sel1.0-l2fp1e-3-withl2.log

python -u trainval_global_fedprox.py \
    --n_iccad2012   2 \
    --n_asml1       2 \
    --sel_ratio     1.0 \
    --model_path    models/model-fedprox-a2i2-sel1.0-l2fp1e-3-withl2 \
    | tee $LOG_DIR/fedprox-a2i2-sel1.0-l2fp1e-3-withl2.log

python -u trainval_global_fedprox.py \
    --n_iccad2012   5 \
    --n_asml1       5 \
    --sel_ratio     1.0 \
    --model_path    models/model-fedprox-a5i5-sel1.0-l2fp1e-3-withl2 \
    | tee $LOG_DIR/fedprox-a5i5-sel1.0-l2fp1e-3-withl2.log

python -u trainval_global_fedprox.py \
    --n_iccad2012   2 \
    --n_asml1       2 \
    --sel_ratio     0.5 \
    --model_path    models/model-fedprox-a2i2-sel0.5-l2fp1e-3-withl2 \
    | tee $LOG_DIR/fedprox-a2i2-sel0.5-l2fp1e-3-withl2.log

python -u trainval_global_fedprox.py \
    --n_iccad2012   5 \
    --n_asml1       5 \
    --sel_ratio     0.5 \
    --model_path    models/model-fedprox-a5i5-sel0.5-l2fp1e-3-withl2 \
    | tee $LOG_DIR/fedprox-a5i5-sel0.5-l2fp1e-3-withl2.log
