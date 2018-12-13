#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "omp.h"

//See values of N in assignment instructions.
//#define N 100000
//Do not change the seed, or your answer will not be correct
#define SEED 72

//For GPU implementation
#define BLOCK_SIZE 1024


struct pointData{
    double x;
    double y;
};


void generateDataset(struct pointData * data);
__global__ void findEpsilon(pointData *data, unsigned int *count, double *epsilon);


int main(int argc, char *argv[]) {
	double t_start;

	//Read epsilon distance from command line
	if (argc!=2) {
	    printf("\nIncorrect number of input parameters. Please input an epsilon distance.\n");
	    return 0;
	}
	
	
	char inputEpsilon[20];
	strcpy(inputEpsilon,argv[1]);
	double epsilon = atof(inputEpsilon);
	
	//generate dataset:
	struct pointData *data;
	data = (struct pointData*) malloc(sizeof(struct pointData)*N);
	//printf("ize of dataset (MiB): %f\n",(2.0*sizeof(double)*N*1.0)/(1024.0*1024.0));
	generateDataset(data);

	omp_set_num_threads(1);

    struct pointData *dev_data;
    
    unsigned int *dev_count;
    double       *dev_epsilon;

    cudaMalloc((struct pointData **) &dev_data, sizeof(struct pointData) * N);
    cudaMalloc((int **) &dev_count, sizeof(unsigned int));
    cudaMalloc((double **) &dev_epsilon, sizeof(double));

    t_start = omp_get_wtime();

    cudaMemcpy(dev_data, data, sizeof(struct pointData) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_epsilon, &epsilon, sizeof(double), cudaMemcpyHostToDevice);

    printf("CPU->GPU: %lf\n", omp_get_wtime() - t_start);
    
    cudaMemset(dev_count, 0, sizeof(unsigned int));

    const unsigned int total_blocks = ceil(N/BLOCK_SIZE);

    t_start = omp_get_wtime();

    findEpsilon<<<total_blocks,1024>>>(dev_data, dev_count, dev_epsilon);

    cudaDeviceSynchronize();

    int *count = (int *)malloc(sizeof(int));
    *count     = 0;

    printf("GPU EXEC: %lf\n", omp_get_wtime() - t_start);
    
    t_start = omp_get_wtime();
    
    cudaMemcpy(count, dev_count, sizeof(int), cudaMemcpyDeviceToHost);

    printf("GPU->CPU: %lf\n", omp_get_wtime() - t_start);

    printf("Points in epsilon: %d\n", *count);

	free(data);
	printf("\n");
	return 0;
}


__global__ void findEpsilon(pointData *data, unsigned int *count, double *epsilon) {
    unsigned int tid=threadIdx.x+ (blockIdx.x*blockDim.x);
    
    if(tid >= N) {
        return;
    }

    pointData p1 = data[tid];

    double epsilon_squared = pow(*epsilon, 2);

    for(int j = 0; j < N; j++) {
        pointData p2 = data[j];
        double dist = pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2);

        if(epsilon_squared >= dist) {
            atomicAdd(count, 1);
        }
    }
}


//Do not modify the dataset generator or you will get the wrong answer
void generateDataset(struct pointData * data) {
	//seed RNG
	srand(SEED);

	for (unsigned int i=0; i<N; i++) {
		data[i].x = 1000.0*((double)(rand()) / RAND_MAX);	
		data[i].y = 1000.0*((double)(rand()) / RAND_MAX);	
	}
}
