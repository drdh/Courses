//*计算π的C语言 MPI编程代码段*// 

#include <mpi.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char *argv[])
{
    int done = 0, n, myid, numprocs, i;
    double PI25DT = 3.141592653589793238462643;
    double mypi, pi, h, sum, x;
    double startwtime = 0.0, endwtime;
    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);  //mpi的初始化
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);  //获取线程数
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);  //获取线程id值
    MPI_Get_processor_name(processor_name, &namelen);  //获取处理器名称

    fprintf(stderr, "Process %d on %s\n", myid, processor_name);

    n = 0;
    while(!done){
        if(myid == 0){
            if(n == 0)
                n = 100;
            else
                n = 0;
            startwtime = MPI_Wtime();
        }
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);  //进行广播传送消息
        if(n == 0)
            done = 1;
        else{
            h = 1.0/(double)n;
            sum = 0.0;
            for(i=myid+1; i<=n; i+=numprocs) { //各线程计算自己的面积
                x = h * ((double)i - 0.5);
                sum += (4.0/(1.0 + x*x));
            }
            mypi = h * sum;
            MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);   //归约，mypi为发送方，pi为接收方
            if(myid == 0){
                printf("pi is approximately %.16f,Error is %.16f\n",pi, fabs(pi-PI25DT));
                endwtime = MPI_Wtime();
                printf("wall clock time = %f\n", endwtime-startwtime);
            }
        }
    }
    MPI_Finalize();   //mpi结束

    return 0;
}