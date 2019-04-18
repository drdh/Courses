#include <stdio.h> 
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define BLOCK_SIZE 4


typedef void (*multiply)(int *A, int *B, int *C,int BLOCK_NUM);

double getTime(int *A, int *B, int *C, multiply mul,int BLOCK_NUM){
    timeval start, finish;
    gettimeofday(&start, 0);
    mul(A, B, C,BLOCK_NUM);
    gettimeofday(&finish, 0);
    double interval = 1e6 * (finish.tv_sec - start.tv_sec) + finish.tv_usec - start.tv_usec;
    return interval;
}

void searial(int *A, int *B, int *C,int BLOCK_NUM){
    for (int i = 0; i < BLOCK_NUM * BLOCK_SIZE; i++){
        for (int j = 0; j < BLOCK_NUM * BLOCK_SIZE; j++){
            int sum = 0;
            for (int k = 0; k < BLOCK_NUM * BLOCK_SIZE; k++){
                sum += A[i * BLOCK_NUM * BLOCK_SIZE + k] * B[k * BLOCK_NUM * BLOCK_SIZE + j];
            }
            C[i * BLOCK_NUM * BLOCK_SIZE + j] = sum;
        }
    }
}

__global__
void deviceParallel1(int *A, int *B, int *C,int BLOCK_NUM){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    for (int i = 0; i < BLOCK_NUM * BLOCK_SIZE; i++){
        sum += A[row * BLOCK_NUM * BLOCK_SIZE + i] * B[i * BLOCK_NUM * BLOCK_SIZE + col];
    }
    C[row * BLOCK_NUM * BLOCK_SIZE + col] = sum;
}

void parallel1(int *A, int *B, int *C,int BLOCK_NUM){
    int *CA, *CB, *CC;
    cudaMalloc(&CA, sizeof(int) * BLOCK_NUM * BLOCK_SIZE * BLOCK_NUM * BLOCK_SIZE);
    cudaMalloc(&CB, sizeof(int) * BLOCK_NUM * BLOCK_SIZE * BLOCK_NUM * BLOCK_SIZE); 
    cudaMalloc(&CC, sizeof(int) * BLOCK_NUM * BLOCK_SIZE * BLOCK_NUM * BLOCK_SIZE);
    cudaMemcpy(CA, A, sizeof(int) * BLOCK_NUM * BLOCK_SIZE * BLOCK_NUM * BLOCK_SIZE, 
                                        cudaMemcpyHostToDevice);

    cudaMemcpy(CB, B, sizeof(int) * BLOCK_NUM * BLOCK_SIZE * BLOCK_NUM * BLOCK_SIZE, 
                                        cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
    dim3 dimGrid(BLOCK_NUM, BLOCK_NUM);
    deviceParallel1<<<dimGrid,dimBlock>>>(CA, CB, CC,BLOCK_NUM);
    cudaDeviceSynchronize();
    cudaMemcpy(C, CC, sizeof(int) * BLOCK_NUM * BLOCK_SIZE * BLOCK_NUM * BLOCK_SIZE, 
                                        cudaMemcpyDeviceToHost);
    cudaFree(CA);
    cudaFree(CB);
    cudaFree(CC);
}

__global__
void deviceParallel2(int *A, int *B, int *C,int BLOCK_NUM){
    //获得线程块号
    int blkRow = blockIdx.y; 
    int blkCol = blockIdx.x;

    //获得块内的线程号 
    int row = threadIdx.y; 
    int col = threadIdx.x;

    int var = 0;

    //循环，遍历所有子矩阵
    for (int i = 0; i < BLOCK_NUM; i++) {   
        const int *ASub = A + blkRow * BLOCK_SIZE * BLOCK_NUM * BLOCK_SIZE + i * BLOCK_SIZE; 
        const int *BSub = B + i * BLOCK_SIZE * BLOCK_NUM * BLOCK_SIZE + blkCol * BLOCK_SIZE;

        __shared__ int Ads[BLOCK_SIZE][BLOCK_SIZE]; 
        __shared__ int Bds[BLOCK_SIZE][BLOCK_SIZE];

        Ads[row][col] = *(ASub + row * BLOCK_SIZE * BLOCK_NUM + col); 
        Bds[row][col] = *(BSub + row * BLOCK_SIZE * BLOCK_NUM + col);

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            var += Ads[row][k] * Bds[k][col]; 
        }
        __syncthreads();
    }
    int *CSub = C + blkRow * BLOCK_SIZE * BLOCK_NUM * BLOCK_SIZE + blkCol * BLOCK_SIZE;
    *(CSub + row * BLOCK_SIZE * BLOCK_NUM + col) = var;
}

void parallel2(int *A, int *B, int *C,int BLOCK_NUM){
    int *CA, *CB, *CC;
    cudaMalloc(&CA, sizeof(int) * BLOCK_NUM * BLOCK_SIZE * BLOCK_NUM * BLOCK_SIZE);
    cudaMalloc(&CB, sizeof(int) * BLOCK_NUM * BLOCK_SIZE * BLOCK_NUM * BLOCK_SIZE); 
    cudaMalloc(&CC, sizeof(int) * BLOCK_NUM * BLOCK_SIZE * BLOCK_NUM * BLOCK_SIZE);

    cudaMemcpy(CA, A, sizeof(int) * BLOCK_NUM * BLOCK_SIZE * BLOCK_NUM * BLOCK_SIZE, 
                                        cudaMemcpyHostToDevice);

    cudaMemcpy(CB, B, sizeof(int) * BLOCK_NUM * BLOCK_SIZE * BLOCK_NUM * BLOCK_SIZE, 
                                        cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
    dim3 dimGrid(BLOCK_NUM, BLOCK_NUM);

    deviceParallel2<<<dimGrid,dimBlock>>>(CA, CB, CC,BLOCK_NUM);

    cudaDeviceSynchronize();

    cudaMemcpy(C, CC, sizeof(int) * BLOCK_NUM * BLOCK_SIZE * BLOCK_NUM * BLOCK_SIZE, 
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
    int BLOCK_NUM=3;

    int *A = new int[BLOCK_NUM * BLOCK_SIZE * BLOCK_NUM * BLOCK_SIZE];
    int *B = new int[BLOCK_NUM * BLOCK_SIZE * BLOCK_NUM * BLOCK_SIZE];
    int *C1 = new int[BLOCK_NUM * BLOCK_SIZE * BLOCK_NUM * BLOCK_SIZE];
    int *C2 = new int[BLOCK_NUM * BLOCK_SIZE * BLOCK_NUM * BLOCK_SIZE];
    int *C3 = new int[BLOCK_NUM * BLOCK_SIZE * BLOCK_NUM * BLOCK_SIZE];

    // 读取矩阵数据
    read(A, BLOCK_NUM * BLOCK_SIZE, BLOCK_NUM * BLOCK_SIZE);
    read(B, BLOCK_NUM * BLOCK_SIZE, BLOCK_NUM * BLOCK_SIZE);

    cout << "Serial Time = " << getTime(A, B, C1, searial,BLOCK_NUM) << " ps." << endl;
    cout << "Parallel1 Time = " << getTime(A, B, C2, parallel1,BLOCK_NUM) << " ps." << endl;
    cout << "Parallel2 Time = " << getTime(A, B, C3, parallel2,BLOCK_NUM) << " ps." << endl;

    
    for(int i=0;i<BLOCK_NUM* BLOCK_SIZE * BLOCK_NUM * BLOCK_SIZE;i++){
        if(C1[i]!=C2[i]|| C1[i]!=C3[i] ){
            cout<<"error "<<i<<" "<<C1[i]<<","<<C2[i]<<","<<C3[i]<<endl;
        }
    }
    
    delete[] A;
    delete[] B;
    delete[] C1;
    delete[] C2;
    delete[] C3;

    return 0;
}
