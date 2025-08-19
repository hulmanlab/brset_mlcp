#!/bin/bash
#SBATCH --account brset_analysis
#SBATCH --mem 256g
#SBATCH -c 64
#SBATCH --partition gpu
#SBATCH --gres=gpu:2
#SBATCH -w gn-1002
#SBATCH --time 6:00:00
cd /home/livieymli/brset_analysis/BRSET
/home/livieymli/miniforge3/envs/cu312/bin/python template_3class_DR_mBRSET_EXEVAL.py -b convnextv2_large -bm fine_tune