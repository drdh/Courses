#include <cuda_runtime.h>
#include <cstdio>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>

using namespace std;

#define BLOCK_SIZE 4

double getTime( int * dA, int *dx,int *dy, int nRows, int nx,
    void (*mul)( int * dA, int *dx,int *dy, int nRows, int nx)){
    timeval start, finish;
    gettimeofday(&start, 0);
    mul(dA,dx,dy,nRows,nx);
    gettimeofday(&finish, 0);
    double interval = 1e6 * (finish.tv_sec - start.tv_sec) + finish.tv_usec - start.tv_usec;
    return interval;
}

void matvect_serial( int * dA, int *dx,int *dy, int nRows, int nx){
    for(int i=0;i<nRows;i++){
        int sum=0;
        for(int j=0;i<nx;j++){
            sum+=dA[i*nx+j]*dx[j];
        }
        dy[i]=sum;
    }
}

__global__ void matvec_kernel(int *dA, int *dx,int *dy, int nRows, int nCols){
    int tid=threadIdx.x + blockIdx.x * blockDim.x;

   __shared__ int x_shared[BLOCK_SIZE];
   int y_val = 0;

   #pragma unroll
   for (unsigned int m = 0; m < ((nCols + BLOCK_SIZE - 1)/ BLOCK_SIZE); ++m)
   {
       if ((m * BLOCK_SIZE + threadIdx.x) <  nCols) 
           x_shared[threadIdx.x] = dx[threadIdx.x + m * BLOCK_SIZE];
       else                                         
           x_shared[threadIdx.x] = 0;

       __syncthreads();

       #pragma unroll
       for (unsigned int e = 0; e < BLOCK_SIZE; ++e) {
           // --- Column-major ordering - faster
           y_val += dA[tid + (e + BLOCK_SIZE * m) * nRows] * x_shared[e];
           // --- Row-major ordering - slower
           //y_val += dA[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
       }
       __syncthreads();
   }
   if (tid < nRows) dy[tid] = y_val;
}


__host__ void matvec( int * dA, int *dx,int *dy, int nRows, int nx){
    int *CdA,*Cdx,*Cdy;
    cudaMalloc(&CdA,sizeof(int)*nRows*nx);
    cudaMalloc(&Cdx,sizeof(int)*nx);
    cudaMalloc(&Cdy,sizeof(int)*nRows);

    cudaMemcpy(CdA,dA,sizeof(int)*nRows*nx,cudaMemcpyHostToDevice);
    cudaMemcpy(Cdx,dx,sizeof(int)*nx,cudaMemcpyHostToDevice);


    dim3 grid((nRows+BLOCK_SIZE-1)/BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    matvec_kernel<<<grid,block>>>(dA,dx,dy,nRows,nx);

    cudaDeviceSynchronize();

    cudaMemcpy(dy,Cdy,sizeof(int)*nRows,cudaMemcpyDeviceToHost);

    cudaFree(CdA);
    cudaFree(Cdx);
    cudaFree(Cdy);
    
}


int main(int argc, char const *argv[]){
    int nx=13;
    int nRows=7;

    int *dA=(int *)malloc(sizeof(int)*nx*nRows);
    int *dx=(int *)malloc(sizeof(int)*nx);
    int *dy=(int *)malloc(sizeof(int)*nRows);
    int *dr=(int *)malloc(sizeof(int)*nRows);

    srand((unsigned)time(NULL));
    for(int i=0;i<nRows;i++){
        for(int j=0;i<nx;j++){
            dA[i*nx+j]=rand()%100;
        }
    }
    for(int i=0;i<nx;i++){
        dx[i]=rand()%100;
    }

    cout << "Serial Time = " << getTime(dA,dx,dr,nRows,nx,matvec) << " ps." << endl;
    cout << "Parallel1 Time = " << getTime(dA,dx,dy,nRows,nx,matvect_serial) << " ps." << endl;

    free(dA);
    free(dx);
    free(dy);
    free(dr);
}