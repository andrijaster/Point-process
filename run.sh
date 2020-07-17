#!/usr/bin/env bash

export DATA_FOLDER="data/autoput/prepared/"
export TRAINING_DATASET="stan1_traka1_01012017.csv"
python3 src/auto_model_training.py
