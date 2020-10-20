#!/bin/sh
#$ -cwd
#$ -l q_node=4
#$ -l h_rt=1:00:00
#$ -N q4-rankdebug-nocache-val100step-aug-hvd-test
. /etc/profile.d/modules.sh
module load python cuda/10.1.105 cudnn openmpi
rm /gs/hs0/tga-yamaguchi.m/ji/*tfcache*
mpirun -n 4 -bind-to none --map-by node -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH  -x NCCL_IB_DISABLE=1 python code/unet-ki67/main.py
