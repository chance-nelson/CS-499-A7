#!/bin/bash
#SBATCH --job-name=cpu-q3           
#SBATCH --output=/home/cbn35/CS-499-A7/Q3/output_cpu	
#SBATCH --time=30:00				         
#SBATCH --workdir=/home/cbn35/CS-499-A7/Q3
#SBATCH --mem=2G
#SBATCH -c4


module load gcc

num='100'
gcc -DN=$num -lm point_epsilon_bruteforce.c -o ep-cpu -fopenmp
printf "$num\n"
./ep-cpu 5
printf '\n'
./ep-cpu 10

num='1000'
gcc -DN=$num -lm point_epsilon_bruteforce.c -o ep-cpu -fopenmp
printf "$num\n"
./ep-cpu 5
printf '\n'
./ep-cpu 10

num='10000'
gcc -DN=$num -lm point_epsilon_bruteforce.c -o ep-cpu -fopenmp
printf "$num\n"
./ep-cpu 5
printf '\n'
./ep-cpu 10

num='100000'
gcc -DN=$num -lm point_epsilon_bruteforce.c -o ep-cpu -fopenmp
printf "$num\n"
./ep-cpu 5
printf '\n'
./ep-cpu 10
