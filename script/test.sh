#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=7  python test_net.py --dataset pascal_voc --net res101 --checksession 1 --checkepoch 20 --checkpoint 5010 --cuda #--lighthead
