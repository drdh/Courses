#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h>
#include "mpi.h"

int i,j,k;
int N = 36;



int cmp(const void * a, const void * b) {
  if (*(int*)a < *(int*)b) return -1;
  if (*(int*)a > *(int*)b) return 1;
  else return 0;
}

void phase1(int *array, int N, int startIndex, int subArraySize, int *pivots, int p) {
  // 对子数组进行局部排序
  qsort(array + startIndex, subArraySize, sizeof(array[0]), cmp);

  // 正则采样
  for (i = 0; i < p; i++) {
    pivots[i] = array[startIndex + (i * (N / (p * p)))];    
  }
  return;
}

void phase2(int *array, int startIndex, int subArraySize, int *pivots, int *partitionSizes, int p, int myId) {
  int *collectedPivots = (int *) malloc(p * p * sizeof(pivots[0]));
  int *phase2Pivots = (int *) malloc((p - 1) * sizeof(pivots[0]));          //主元
  int index = 0;

  //收集消息，根进程在它的接受缓冲区中包含所有进程的发送缓冲区的连接。
  MPI_Gather(pivots, p, MPI_INT, collectedPivots, p, MPI_INT, 0, MPI_COMM_WORLD);       
  if (myId == 0) {

    qsort(collectedPivots, p * p, sizeof(pivots[0]), cmp);          //对正则采样的样本进行排序

    // 采样排序后进行主元的选择
    for (i = 0; i < (p -1); i++) {
      phase2Pivots[i] = collectedPivots[(((i+1) * p) + (p / 2)) - 1];
    }
  }
  //发送广播
  MPI_Bcast(phase2Pivots, p - 1, MPI_INT, 0, MPI_COMM_WORLD);
  // 进行主元划分，并计算划分部分的大小
  for ( i = 0; i < subArraySize; i++) {
    if (array[startIndex + i] > phase2Pivots[index]) {
      //如果当前位置的数字大小超过主元位置，则进行下一个划分
      index += 1;
    }
    if (index == p) {
      //最后一次划分，子数组总长减掉当前位置即可得到最后一个子数组划分的大小
      partitionSizes[p - 1] = subArraySize - i + 1;
      break;
    }
    partitionSizes[index]++ ;   //划分大小自增
  }
  free(collectedPivots);
  free(phase2Pivots);
  return;
}

void phase3(int *array, int startIndex, int *partitionSizes, int **newPartitions, int *newPartitionSizes, int p) {
  int totalSize = 0;
  int *sendDisp = (int *) malloc(p * sizeof(int));
  int *recvDisp = (int *) malloc(p * sizeof(int));

  // 全局到全局的发送，每个进程可以向每个接收者发送数目不同的数据.
  MPI_Alltoall(partitionSizes, 1, MPI_INT, newPartitionSizes, 1, MPI_INT, MPI_COMM_WORLD);

  // 计算划分的总大小，并给新划分分配空间
  for ( i = 0; i < p; i++) {
    totalSize += newPartitionSizes[i];
  }
  *newPartitions = (int *) malloc(totalSize * sizeof(int));

  // 在发送划分之前计算相对于sendbuf的位移，此位移处存放着输出到进程的数据
  sendDisp[0] = 0;
  recvDisp[0] = 0;      //计算相对于recvbuf的位移，此位移处存放着从进程接受到的数据
  for ( i = 1; i < p; i++) {
    sendDisp[i] = partitionSizes[i - 1] + sendDisp[i - 1];
    recvDisp[i] = newPartitionSizes[i - 1] + recvDisp[i - 1];
  }

  //发送数据，实现n次点对点通信
  MPI_Alltoallv(&(array[startIndex]), partitionSizes, sendDisp, MPI_INT, *newPartitions, newPartitionSizes, recvDisp, MPI_INT, MPI_COMM_WORLD);

  free(sendDisp);
  free(recvDisp);
  return;
}

void phase4(int *partitions, int *partitionSizes, int p, int myId, int *array) {
    int *sortedSubList;
    int *recvDisp, *indexes, *partitionEnds, *subListSizes, totalListSize;

    indexes = (int *) malloc(p * sizeof(int));
    partitionEnds = (int *) malloc(p * sizeof(int));
    indexes[0] = 0;
    totalListSize = partitionSizes[0];
    for ( i = 1; i < p; i++) {
      totalListSize += partitionSizes[i];
      indexes[i] = indexes[i-1] + partitionSizes[i-1];
      partitionEnds[i-1] = indexes[i];
    }
    partitionEnds[p - 1] = totalListSize;

    sortedSubList = (int *) malloc(totalListSize * sizeof(int));
    subListSizes = (int *) malloc(p * sizeof(int));
    recvDisp = (int *) malloc(p * sizeof(int));

    // 归并排序
    for ( i = 0; i < totalListSize; i++) {
      int lowest = INT_MAX;
      int ind = -1;
      for (j = 0; j < p; j++) {
        if ((indexes[j] < partitionEnds[j]) && (partitions[indexes[j]] < lowest)) {
      lowest = partitions[indexes[j]];
      ind = j;
        }
      }
      sortedSubList[i] = lowest;
      indexes[ind] += 1;
    }

    // 发送各子列表的大小回根进程中
    MPI_Gather(&totalListSize, 1, MPI_INT, subListSizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 计算根进程上的相对于recvbuf的偏移量
    if (myId == 0) {
      recvDisp[0] = 0;
      for ( i = 1; i < p; i++) {
        recvDisp[i] = subListSizes[i - 1] + recvDisp[i - 1];
      }
    }

    //发送各排好序的子列表回根进程中
    MPI_Gatherv(sortedSubList, totalListSize, MPI_INT, array, subListSizes, recvDisp, MPI_INT, 0, MPI_COMM_WORLD);

    free(partitionEnds);
    free(sortedSubList);
    free(indexes);
    free(subListSizes);
    free(recvDisp);
    return;
}

//PSRS排序函数，调用了4个过程函数
void psrs_mpi(int *array, int N)    
{
      int p, myId, *partitionSizes, *newPartitionSizes, nameLength;
      int subArraySize, startIndex, endIndex, *pivots, *newPartitions;
      char processorName[MPI_MAX_PROCESSOR_NAME];


      MPI_Comm_size(MPI_COMM_WORLD,&p);
      MPI_Comm_rank(MPI_COMM_WORLD,&myId);
      MPI_Get_processor_name(processorName,&nameLength);

      printf("Process %d is on %s\n",myId, processorName);

      pivots = (int *) malloc(p*sizeof(int));
      partitionSizes = (int *) malloc(p*sizeof(int));
      newPartitionSizes = (int *) malloc(p*sizeof(int));
      for ( k = 0; k < p; k++) {
        partitionSizes[k] = 0;
      }

      // 获取起始位置和子数组大小
      startIndex = myId * N / p;
      if (p == (myId + 1)) {
        endIndex = N;
      } 
      else {
        endIndex = (myId + 1) * N / p;
      }
      subArraySize = endIndex - startIndex;

      MPI_Barrier(MPI_COMM_WORLD);
      //调用各阶段函数
      phase1(array, N, startIndex, subArraySize, pivots, p);
      if (p > 1) {
        phase2(array, startIndex, subArraySize, pivots, partitionSizes, p, myId);
        phase3(array, startIndex, partitionSizes, &newPartitions, newPartitionSizes, p);
        phase4(newPartitions, newPartitionSizes, p, myId, array);
      }

      if (myId == 0) 
      for(k = 0; k < N; k++){
          printf("%d ",array[k]);
      }
      printf("\n");
      if (p > 1) {
        free(newPartitions);
      }
      free(partitionSizes);
      free(newPartitionSizes);
      free(pivots);


    free(array);
    MPI_Finalize();

  }

  int main(int argc, char *argv[]) {

    int *array;
    array = (int *) malloc(N*sizeof(int));

      srand(100);
      for ( k = 0; k < N; k++) {
        array[k] = rand()%100;
      }
      MPI_Init(&argc,&argv);      //MPI初始化
      psrs_mpi(array,N);          //调用PSRS算法进行并行排序

    return 0;
  }