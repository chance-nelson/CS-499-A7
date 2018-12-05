#!/bin/bash
#SBATCH --job-name=gpu-detect           
#SBATCH --output=/scratch/cbn35/CS-499-A7/output	
#SBATCH --time=5:00				         
#SBATCH --workdir=/scratch/cbn35/CS-499-A7
#SBATCH --mem=100M
#SBATCH --gres=gpu
#SBATCH --qos=gpu_class

srun date
srun hostname
srun nvidia-smi
