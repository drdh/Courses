//使用并行域并行化的程序
#include <stdio.h>
#include <omp.h>
static long num_steps = 100000;
double step;
#define NUM_THREADS 2

int main ()
{ 
    int i;
    double x, pi, sum[NUM_THREADS];
    step = 1.0/(double) num_steps;
    omp_set_num_threads(NUM_THREADS);  //设置2线程
 #pragma omp parallel private(i)  //并行域开始，每个线程(0和1)都会执行该代码
{
    double x;
    int id;
    id = omp_get_thread_num();
    for (i=id, sum[id]=0.0;i< num_steps; i=i+NUM_THREADS){
        x = (i+0.5)*step;
        sum[id] += 4.0/(1.0+x*x);
    }
}
    for(i=0, pi=0.0;i<NUM_THREADS;i++)  pi += sum[i] * step;
    printf("%lf\n",pi);
 }  
//共2个线程参加计算，其中线程0进行迭代步0,2,4,...线程1进行迭代步1,3,5,....
