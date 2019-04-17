#include<cstdio>
#include<cstdlib>
#include "mpi.h"
#include <limits.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h>

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
        int *collected_pivot=(int *)malloc(thread_num*thread_num*sizeof(pivot[0]));
        int *final_pivot=(int *)malloc((thread_num-1)*sizeof(pivot[0]));
        //0进程收集各个进程得到的主元
        MPI_Gather(pivot,thread_num,MPI_INT,collected_pivot,thread_num,MPI_INT,0,MPI_COMM_WORLD);
        if(ID==0){
            qsort(collected_pivot,thread_num*thread_num,sizeof(pivot[0]),cmp);//样本排序
            for(int i=0;i<thread_num-1;i++){//选择主元
                final_pivot[i]=collected_pivot[(((i+1) * thread_num) + (thread_num / 2)) - 1];
            }
        }
        MPI_Bcast(final_pivot,thread_num-1,MPI_INT,0,MPI_COMM_WORLD);
        int index=0;
        int *partition_size=(int *)malloc(thread_num*sizeof(int));
        for(int i=0;i<thread_num;i++){
            partition_size[i]=0;
        }
        for(int i=0;i<sub_array_size;i++){//主元划分
            if(array[start_index+i]>final_pivot[index]){
                index++;
            }
            if(index==thread_num){
                partition_size[thread_num-1]=sub_array_size-i+1;
                break;
            }
            partition_size[index]++;
        }
        free(collected_pivot);
        free(final_pivot);

        int *new_partition_size=(int *)malloc(thread_num*sizeof(int));
        MPI_Alltoall(partition_size,1,MPI_INT,new_partition_size,1,MPI_INT,MPI_COMM_WORLD);
        int total_size=0;
        for(int i=0;i<thread_num;i++){
            total_size+=new_partition_size[i];
        }
        int *new_partition=(int *)malloc(total_size*sizeof(int));

        int *send_disp = (int *) malloc(thread_num * sizeof(int));
        int *recv_disp = (int *) malloc(thread_num * sizeof(int));
        int *recv_ends = (int *) malloc(thread_num * sizeof(int));
        send_disp[0]=0;recv_disp[0]=0;
        for(int i=1;i<thread_num;i++){
            send_disp[i]=partition_size[i-1]+send_disp[i-1];
            recv_disp[i]=new_partition_size[i-1]+recv_disp[i-1];
            recv_ends[i-1]=recv_disp[i];
        }
        recv_ends[thread_num-1]=total_size;
        //全局交换
        MPI_Alltoallv(&(array[start_index]),partition_size,send_disp,MPI_INT,new_partition,new_partition_size,recv_disp,MPI_INT,MPI_COMM_WORLD);
        free(send_disp);
        free(partition_size);
        free(new_partition_size);


        int *sorted_sub_array=(int*)malloc(total_size*sizeof(int));
        //归并排序
        for(int i=0;i<total_size;i++){
            int lowest=INT_MAX;
            int ind=-1;
            for(int j=0;j<thread_num;j++){
                if((recv_disp[j]<recv_ends[j]) && (new_partition[recv_disp[j]]<lowest)){
                    lowest=new_partition[recv_disp[j]];
                    ind=j;
                }
            }
            sorted_sub_array[i]=lowest;
            recv_disp[ind]++;
        }

        int *sub_array_size=(int *)malloc(thread_num*sizeof(int));
        // 发送各子列表的大小回根进程中
        MPI_Gather(&total_size,1,MPI_INT,sub_array_size,1,MPI_INT,0,MPI_COMM_WORLD);

        // 计算根进程上的相对于recvbuf的偏移量
        if(ID==0){
            recv_disp[0]=0;
            for(int i=1;i<thread_num;i++){
                recv_disp[i]=sub_array_size[i-1]+recv_disp[i-1];
            }
        }

        //发送各排好序的子列表回根进程中
        MPI_Gatherv(sorted_sub_array,total_size,MPI_INT,array,sub_array_size,recv_disp,MPI_INT,0,MPI_COMM_WORLD);

        free(recv_disp);
        free(recv_ends);
        free(sorted_sub_array);
        free(sub_array_size);
    }

    if(ID==0){
        for(int i=0;i<N;i++){
            printf("%d ",array[i]);
        }
    }
    printf("\n");
    free(pivot);

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