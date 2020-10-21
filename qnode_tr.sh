#!/bin/sh
#$ -cwd
#$ -l q_node=16
#$ -l h_rt=2:00:00
#$ -N q16-hdf5-noaug-nocache-noshard
. /etc/profile.d/modules.sh
module load python cuda/10.1.105 cudnn openmpi
rm /gs/hs0/tga-yamaguchi.m/ji/*tfcache*
mpirun -n 16 -bind-to none --map-by node -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH  -x NCCL_IB_DISABLE=1 python code/unet-ki67/main.py
