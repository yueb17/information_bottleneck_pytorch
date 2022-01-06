#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python main.py --arch resnet56 --dataset cifar100 \
--method L1 --stage_pr [0,0.7,0.7,0.7,0] --batch_size 128 --wd 0.0005 \
--lr_ft 0:0.1,100:0.01,150:0.001 --epochs 200 --wg weight --pick_pruned rand \
--project rdm_pr0.7_rsn56_cf100 \
--base_model_path Experiments/199192_pretrain_resnet56_cifar100_SERVER-20210926-111829/weights/checkpoint_best.pth