#!/bin/bash
#SBATCH --job-name=gpu-q3           
#SBATCH --array=0-2
#SBATCH --output=/home/cbn35/CS-499-A7/Q3/output_gpu_bf_%a
#SBATCH --time=1:00				         
#SBATCH --workdir=/home/cbn35/CS-499-A7/Q3
#SBATCH --mem=2G
#SBATCH --qos=gpu_class
#SBATCH --constraint=k80


module load cuda

binary=ep_$SLURM_ARRAY_TASK_ID

nvcc -arch=compute_37 -code=sm_37 -lcuda point_epsilon_bruteforce.cu -o $binary --compiler-options='-DN=100 -fopenmp -lm'
printf "100\n"
./$binary 5
printf '\n'
./$binary 10

nvcc -arch=compute_37 -code=sm_37 -lcuda point_epsilon_bruteforce.cu -o $binary --compiler-options='-DN=1000 -fopenmp -lm'
printf "1000\n"
./$binary 5
printf '\n'
./$binary 10

nvcc -arch=compute_37 -code=sm_37 -lcuda point_epsilon_bruteforce.cu -o $binary --compiler-options='-DN=10000 -fopenmp -lm'
printf "10000\n"
./$binary 5
printf '\n'
./$binary 10

nvcc -arch=compute_37 -code=sm_37 -lcuda point_epsilon_bruteforce.cu -o $binary --compiler-options='-DN=100000 -fopenmp -lm'
printf "100000\n"
./$binary 5
printf '\n'
./$binary 10
