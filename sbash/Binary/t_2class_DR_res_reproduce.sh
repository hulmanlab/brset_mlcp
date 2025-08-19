#!/bin/bash
#SBATCH --account brset_analysis
#SBATCH --mem 256g
#SBATCH -c 16
#SBATCH --partition gpu
#SBATCH --gres=gpu:2
#SBATCH --time 72:00:00
cd /home/livieymli/brset_analysis/BRSET
/home/livieymli/miniforge3/envs/cu39/bin/python 2class_DR_templet.py -b resnet50 -bm fine_tune -r True