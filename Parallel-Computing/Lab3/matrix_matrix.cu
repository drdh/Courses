#include <stdio.h> 
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define BLOCK_SIZE 4


typedef void (*multiply)(int *A, int *B, int *C,int width);

double getTime(int *A, int *B, int *C, multiply mul,int width){
    timeval start, finish;
    gettimeofday(&start, 0);
    mul(A, B, C,width);
    gettimeofday(&finish, 0);
    double interval = 1e6 * (finish.tv_sec - start.tv_sec) + finish.tv_usec - start.tv_usec;
    return interval;
}

void searial(int *A, int *B, int *C,int width){
    for (int i = 0; i < width * BLOCK_SIZE; i++){
        for (int j = 0; j < width * BLOCK_SIZE; j++){
            int sum = 0;
            for (int k = 0; k < width * BLOCK_SIZE; k++){
                sum += A[i * width * BLOCK_SIZE + k] * B[k * width * BLOCK_SIZE + j];
            }
            C[i * width * BLOCK_SIZE + j] = sum;
        }
    }
}

__global__
void deviceParallel1(int *A, int *B, int *C,int width){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    for (int i = 0; i < width * BLOCK_SIZE; i++){
        sum += A[row * width * BLOCK_SIZE + i] * B[i * width * BLOCK_SIZE + col];
    }
    C[row * width * BLOCK_SIZE + col] = sum;
}

void parallel1(int *A, int *B, int *C,int width){
    int *CA, *CB, *CC;
    cudaMalloc(&CA, sizeof(int) * width * BLOCK_SIZE * width * BLOCK_SIZE);
    cudaMalloc(&CB, sizeof(int) * width * BLOCK_SIZE * width * BLOCK_SIZE); 
    cudaMalloc(&CC, sizeof(int) * width * BLOCK_SIZE * width * BLOCK_SIZE);
    cudaMemcpy(CA, A, sizeof(int) * width * BLOCK_SIZE * width * BLOCK_SIZE, 
                                        cudaMemcpyHostToDevice);

    cudaMemcpy(CB, B, sizeof(int) * width * BLOCK_SIZE * width * BLOCK_SIZE, 
                                        cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
    dim3 dimGrid(width, width);
    deviceParallel1<<<dimBlock, dimGrid>>>(CA, CB, CC,width);
    //cudaThreadSynchronize();
    cudaDeviceSynchronize();
    cudaMemcpy(C, CC, sizeof(int) * width * BLOCK_SIZE * width * BLOCK_SIZE, 
                                        cudaMemcpyDeviceToHost);
    cudaFree(CA);
    cudaFree(CB);
    cudaFree(CC);
}

__global__
void deviceParallel2(int *A, int *B, int *C,int width){
    //获得线程块号
    int blkRow = blockIdx.y; 
    int blkCol = blockIdx.x;

    //获得块内的线程号 
    int row = threadIdx.y; 
    int col = threadIdx.x;

    int var = 0;

    //循环，遍历所有子矩阵
    for (int i = 0; i < width; i++) {   
        const int *ASub = A + blkRow * BLOCK_SIZE * width + i * BLOCK_SIZE; 
        const int *BSub = B + i * BLOCK_SIZE * width + blkCol * BLOCK_SIZE;

        __shared__ int Ads[BLOCK_SIZE][BLOCK_SIZE]; 
        __shared__ int Bds[BLOCK_SIZE][BLOCK_SIZE];

        Ads[row][col] = *(ASub + row * BLOCK_SIZE * width + col); 
        Bds[row][col] = *(BSub + row * BLOCK_SIZE * width + col);

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++) {
            var += Ads[row][i] * Bds[i][col]; 
        }
        __syncthreads();
    }
    int *CSub = C + blkRow * BLOCK_SIZE * width + blkCol * BLOCK_SIZE;
    *(CSub + row * BLOCK_SIZE * width + col) = var;
}

void parallel2(int *A, int *B, int *C,int width){
    int *CA, *CB, *CC;
    cudaMalloc(&CA, sizeof(int) * width * BLOCK_SIZE * width * BLOCK_SIZE);
    cudaMalloc(&CB, sizeof(int) * width * BLOCK_SIZE * width * BLOCK_SIZE); 
    cudaMalloc(&CC, sizeof(int) * width * BLOCK_SIZE * width * BLOCK_SIZE);

    cudaMemcpy(CA, A, sizeof(int) * width * BLOCK_SIZE * width * BLOCK_SIZE, 
                                        cudaMemcpyHostToDevice);

    cudaMemcpy(CB, B, sizeof(int) * width * BLOCK_SIZE * width * BLOCK_SIZE, 
                                        cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
    dim3 dimGrid(width, width);

    deviceParallel2<<<dimBlock, dimGrid>>>(CA, CB, CC,width);

    //cudaThreadSynchronize();
    cudaDeviceSynchronize();

    cudaMemcpy(C, CC, sizeof(int) * width * BLOCK_SIZE * width * BLOCK_SIZE, 
                                        cudaMemcpyDeviceToHost);

    cudaFree(CA);
    cudaFree(CB);
    cudaFree(CC);
}

void read(int *M, int row, int col){
    srand((unsigned)time(NULL));
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            M[i * col + j] = rand() % 100;
        }
    }
}


int main(int argc, char const *argv[]){
    int width=4;

    int *A = new int[width * BLOCK_SIZE * width * BLOCK_SIZE];
    int *B = new int[width * BLOCK_SIZE * width * BLOCK_SIZE];
    int *C1 = new int[width * BLOCK_SIZE * width * BLOCK_SIZE];
    int *C2 = new int[width * BLOCK_SIZE * width * BLOCK_SIZE];
    int *C3 = new int[width * BLOCK_SIZE * width * BLOCK_SIZE];

    // 读取矩阵数据
    read(A, width * BLOCK_SIZE, width * BLOCK_SIZE);
    read(B, width * BLOCK_SIZE, width * BLOCK_SIZE);

    cout << "Serial Time = " << getTime(A, B, C1, searial,width) << " ps." << endl;
    cout << "Parallel1 Time = " << getTime(A, B, C2, parallel1,width) << " ps." << endl;
    //cout << "Parallel2 Time = " << getTime(A, B, C3, parallel2,width) << " ps." << endl;

    
    for(int i=0;i<width* BLOCK_SIZE * width * BLOCK_SIZE;i++){
        if(C1[i]!=C2[i]|| C1[i]!=C3[i] ){
            cout<<"error "<<C1[i]<<","<<C2[i]<<","<<C3[i]<<endl;
        }
    }
    
    delete[] A;
    delete[] B;
    delete[] C1;
    delete[] C2;
    delete[] C3;

    return 0;
}
