#!/bin/bash
#SBATCH --job-name=gpu-q2           
#SBATCH --output=/scratch/cbn35/CS-499-A7/Q2/output	
#SBATCH --time=00:20				         
#SBATCH --workdir=/scratch/cbn35/CS-499-A7/Q2
#SBATCH --mem=5G
#SBATCH --gres=gpu
#SBATCH --qos=gpu_class
#SBATCH --constraint=k80


module load cuda

nvcc -arch=compute_37 -code=sm_37 -lcuda vector_add.cu -o va --compiler-options='-DN=100 -fopenmp'
./va

printf '\n'
nvcc -arch=compute_37 -code=sm_37 -lcuda vector_add.cu -o va --compiler-options='-DN=1000 -fopenmp'
./va

printf '\n'
nvcc -arch=compute_37 -code=sm_37 -lcuda vector_add.cu -o va --compiler-options='-DN=1000000 -fopenmp'
./va

printf '\n'
nvcc -arch=compute_37 -code=sm_37 -lcuda vector_add.cu -o va --compiler-options='-DN=100000000 -fopenmp'
./va

printf '\n'
nvcc -arch=compute_37 -code=sm_37 -lcuda vector_add.cu -o va --compiler-options='-DN=500000000 -fopenmp'
./va
