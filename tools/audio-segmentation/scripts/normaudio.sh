#!/bin/bash

set -e

CONFIG_FILE='./configs/ein_seld/seld.yaml'

python seld/NormAudio.py -c $CONFIG_FILE infer --num_workers=8