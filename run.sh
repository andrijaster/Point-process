#!/usr/bin/env bash

export DATA_FOLDER="data/autoput/prepared/"
export TRAINING_DATASET="stan1_traka1_7-17_04072017.csv"
python3.7 src/auto_model_training.py
