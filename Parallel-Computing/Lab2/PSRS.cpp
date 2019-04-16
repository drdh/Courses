#include<cstdio>
#include<cstdlib>
#include "mpi.h"

int cmp(const void *a,const void *b){
    if(*(int *)a < *(int *)b)
        return -1;
    else if(*(int *)a > *(int *)b)
        return 1;
    else
        return 0;
}

void PSRS(int array[],int N){
    int thread_num,ID,name_length;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Comm_size(MPI_COMM_WORLD,&thread_num);
    MPI_Comm_rank(MPI_COMM_WORLD,&ID);
    MPI_Get_processor_name(processor_name,&name_length);

    int *pivot=(int *)malloc(thread_num*sizeof(int));
    int start_index,end_index;
    start_index=ID*N/thread_num;
    if(thread_num==ID+1){
        end_index=N;
    }
    else{
        end_index=(ID+1)*N/thread_num;
    }
    int sub_array_size=end_index-start_index;

    //MPI_Barrier(MPI_COMM_WORLD);

    qsort(array+start_index,sub_array_size,sizeof(array[0]),cmp);//局部排序
    
    for(int i=0;i<thread_num;i++){//正则采样
        pivot[i]=array[start_index+(i*(N/(thread_num*thread_num)))];
    }

    if(thread_num>1){
        
    }

    MPI_Finalize();
}

int main(int argc,char *argv[]){
    int N=36;
    int array[N];
    
    srand(100);
    for(int i=0;i<N;i++){
        array[i]=rand()%100;
    }
    MPI_Init(&argc,&argv); //MPI初始化
    PSRS(array,N);
}