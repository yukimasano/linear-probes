#!/bin/bash
# run this code to evaluate your alexnet
# it trains and evaluates 1-crops during training for the last 5 epochs
# and finally invokes the code again for a final 10-crop evaluation.


device=0
DIR="data/ILSVRC12"


EXP=PATH/TO/EXPERIMENTDIR
CKPT=${EXP}/checkpoints/checkpoint400.pth
mkdir -p ${EXP}/checkpointsLP

touch ${EXP}/LP_train.txt
${PYTHON} eval_linear_probes.py \
  --batch-size=192 \
  --epochs=36 \
  --learning-rate=0.01 \
  --workers=8 \
  --tencrops=F \
  --modelpath= ${CKPT} \
  --imagenet-path=${DIR} \
  --ckpt-dir=${EXP}/checkpointsLP/ \
  --data=Imagenet \
  --device=$device | tee -a ${EXP}/LP_train.txt

# finally evaluate with tencrops once:
${PYTHON} eval_linear_probes.py\
  --batch-size=256 \
  --workers=8 \
  --tencrops=True \
  --modelpath=${EXP}/checkpoints/checkpoint.pth \
  --imagenet-path=${DIR} \
  --ckpt-dir=${EXP}/checkpointsLP \
  --data=Imagenet \
  --device=$device \
  --evaluate | tee -a ${EXP}/LP_train.txt