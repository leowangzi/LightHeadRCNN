#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=2  python trainval_net.py --dataset pascal_voc --net res101 --bs 2 --nw 8 --lr 7e-4 --lr_decay_step 30 --epochs 30 --cuda --lighthead
