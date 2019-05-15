#include <stdio.h> 
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <cuda_runtime.h>
#include<vector>

using namespace std;

#define BLOCK_NUM 256
#define BLOCK_SIZE 256

typedef void (*add_func)(int *A,int *B,int *C,int length);

double getTime(int *A,int *B,int *C,add_func add,int length){
    timeval start,finish;
    gettimeofday(&start,0);
    add(A,B,C,length);
    gettimeofday(&finish,0);
    return 1e6 * (finish.tv_sec - start.tv_sec) + finish.tv_usec - start.tv_usec;
}

void serial(int *A,int *B,int *C,int length){
    for(int i=0;i<length;i++){
        C[i]=A[i]+B[i];
    }
}

__global__
void deviceParallel(int *A,int *B,int *C,int length){
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    int stride=blockDim.x*gridDim.x;
    for(int i=index;i<length;i+=stride){
        C[i]=A[i]+B[i];
    }
}

void parallel(int *A,int *B,int *C,int length){
    int *CA,*CB,*CC;
    cudaError_t err = cudaSuccess;

    err =cudaMalloc(&CA,sizeof(int)*length);
    err =cudaMalloc(&CB,sizeof(int)*length);
    err =cudaMalloc(&CC,sizeof(int)*length);

    if(err != cudaSuccess)
    {
        printf("the cudaMalloc on GPU is failed");
    }

    cudaMemcpy(CA,A,sizeof(int)*length,cudaMemcpyHostToDevice);
    cudaMemcpy(CB,B,sizeof(int)*length,cudaMemcpyHostToDevice);
    
    deviceParallel<<<BLOCK_NUM,BLOCK_SIZE>>>(CA,CB,CC,length);
    cudaDeviceSynchronize();
    cudaMemcpy(C,CC,sizeof(int)*length,cudaMemcpyDeviceToHost);
    cudaFree(CA);
    cudaFree(CB);
    cudaFree(CC);
}

void read(int *A,int length){
    srand((unsigned)time(NULL));
    for(int i=0;i<length;i++){
        A[i]=rand()%100;
    }
}

int main(){
    vector<double>size1={1e5,2e5,1e6,2e6,1e7,2e7};
    vector<int>size;
    for(auto i:size1){
        size.push_back(int(i));
    }
    for(int length:size){
        int *A=new int[length];
        int *B=new int[length];
        int *C1=new int[length];
        int *C2=new int[length];

        read(A,length);
        read(B,length);

        double time_1=getTime(A,B,C1,serial,length);
        double time_2=getTime(A,B,C2,parallel,length);

        cout<<"\nLength:"<<length<<endl;
        cout<<"Serial Time: "<<time_1<<"ps."<<endl;
        cout<<"Parallel Time: "<<time_2<<"ps."<<endl;
        cout<<"Speedup: "<<time_1/time_2<<"ps."<<endl;

        int count=0;
        for(int i=0;i<length;i++){
            if(C1[i]!=C2[i]){
                cout<<"error "<<i<<":"<<C1[i]<<","<<C2[i]<<endl;
                count++;
                if(count>10)
                    break;
            }
        }
        delete[]A;
        delete[]B;
        delete[]C1;
        delete[]C2;
    }  
    
}