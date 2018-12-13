#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <omp.h>


//#define N 100 


using namespace std;


void warmUpGPU();
__global__ void vectorAdd(unsigned int * A, unsigned int * B, unsigned int * C);


int main(int argc, char *argv[]) {
    printf("RUNNING FOR N=%d\n", N);
    
    double t_start;

    warmUpGPU();
	
	unsigned int * A;
	unsigned int * B;
	unsigned int * C;
	unsigned int * C_CPU;

	A     = (unsigned int *) malloc(sizeof(unsigned int)*N);
	B     = (unsigned int *) malloc(sizeof(unsigned int)*N);
	C     = (unsigned int *) malloc(sizeof(unsigned int)*N);
	C_CPU = (unsigned int *) malloc(sizeof(unsigned int)*N);


	printf("Size of A+B+C (GiB): %f\n",(sizeof(unsigned int)*N*3.0)/(1024.0*1024.0*1024.0));
	
	
    //init:
	int i = 0;
	for(i = 0; i < N; i++) {
		A[i]     = i;
		B[i]     = i;
		C[i]     = 0;
		C_CPU[i] = 0;
	}

	//CPU version:
	
    t_start = omp_get_wtime();
	
    for(int i = 0; i < N; i++) {
		C_CPU[i] = A[i] + B[i];
	}
	
    printf("CPU TIME: %lf\n", omp_get_wtime() - t_start);
	
	//CUDA error code:
	cudaError_t errCode = cudaSuccess;

	unsigned int * dev_A;
	unsigned int * dev_B;
	unsigned int * dev_C;

	//allocate on the device: A, B, C
	errCode = cudaMalloc((unsigned int**)&dev_A, sizeof(unsigned int)*N);	
	if(errCode != cudaSuccess) {
	    cout << "\nError: A error with code " << errCode << endl; 
	}

	errCode = cudaMalloc((unsigned int**)&dev_B, sizeof(unsigned int)*N);	
	if(errCode != cudaSuccess) {
	    cout << "\nError: B error with code " << errCode << endl; 
	}

	errCode = cudaMalloc((unsigned int**)&dev_C, sizeof(unsigned int)*N);	
	if(errCode != cudaSuccess) {
	    cout << "\nError: C error with code " << errCode << endl; 
	}

    t_start = omp_get_wtime();
	
    //copy A to device
	errCode = cudaMemcpy(dev_A, A, sizeof(unsigned int)*N, cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	    cout << "\nError: A memcpy error with code " << errCode << endl; 
	}	
	
	//copy B to device
	errCode = cudaMemcpy(dev_B, B, sizeof(unsigned int)*N, cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	    cout << "\nError: A memcpy error with code " << errCode << endl; 
	}

	//copy C to device (initialized to 0)
	errCode = cudaMemcpy(dev_C, C, sizeof(unsigned int)*N, cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	    cout << "\nError: A memcpy error with code " << errCode << endl; 
	}

    printf("CPU->GPU COPY TIME: %lf\n", omp_get_wtime() - t_start);

	//execute kernel
	const unsigned int totalBlocks=ceil(N*1.0/1024.0);
	
    printf("total blocks: %d\n",totalBlocks);

    t_start = omp_get_wtime();    
    
    vectorAdd<<<totalBlocks,1024>>>(dev_A, dev_B, dev_C);

    cudaDeviceSynchronize();

    printf("GPU RUN TIME: %lf\n", omp_get_wtime() - t_start);

	if(errCode != cudaSuccess) {
		cout<<"Error after kernel launch "<<errCode<<endl;
	}


    t_start = omp_get_wtime();

	//copy data from device to host 
	errCode=cudaMemcpy( C, dev_C, sizeof(unsigned int)*N, cudaMemcpyDeviceToHost);
	if(errCode != cudaSuccess) {
	    cout << "\nError: getting C result form GPU error with code " << errCode << endl; 
	}

    printf("GPU->CPU COPY TIME: %lf\n", omp_get_wtime() - t_start);
	
    return 0;
}


__global__ void vectorAdd(unsigned int * A, unsigned int * B, unsigned int * C) {
    unsigned int tid=threadIdx.x+ (blockIdx.x*blockDim.x); 

    if (tid>=N){
        return;
    }

    C[tid]=A[tid]+B[tid];

    return;
}


__global__ void warmup(unsigned int * tmp) {
    if (threadIdx.x==0) *tmp = 555;

    return;
}


void warmUpGPU() {
    printf("Warming up GPU for time trialing...\n");	
    
    unsigned int *dev_tmp;
    unsigned int *tmp;
    
    tmp  = (unsigned int*) malloc(sizeof(unsigned int));
    *tmp = 0;
    
    cudaError_t errCode = cudaSuccess;
    
    errCode = cudaMalloc((unsigned int**) &dev_tmp, sizeof(unsigned int));	
    
    if(errCode != cudaSuccess) {
        cout << "\nError: dev_tmp error with code " << errCode << endl; 
    }

    warmup<<<1,256>>>(dev_tmp);

    //copy data from device to host 
    errCode=cudaMemcpy(tmp, dev_tmp, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if(errCode != cudaSuccess) {
        cout << "\nError: getting tmp result form GPU error with code " << errCode << endl; 
    }

    cudaDeviceSynchronize();

    printf("\ntmp (changed to 555 on GPU): %d", *tmp);

    cudaFree(dev_tmp);

    return;
}
