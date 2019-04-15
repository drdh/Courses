//使用并行规约的并行程序
#include <stdio.h>
#include <omp.h>
static long num_steps = 100000;
double step;
#define NUM_THREADS 2

int main ()
{ 
    int i;
    double pi=0.0;
    double sum=0.0;
    double x=0.0;
    step = 1.0/(double) num_steps;
    omp_set_num_threads(NUM_THREADS);  //设置2线程
#pragma omp parallel for reduction(+:sum) private(x) //每个线程保留一份私有拷贝sum，x为线程私有，最后对线程中所以sum进行+规约，并更新sum的全局值
    for(i=1;i<= num_steps; i++){
        x = (i-0.5)*step;
        sum += 4.0/(1.0+x*x);
    }
    pi = sum * step;
    printf("%lf\n",pi);
}   //共2个线程参加计算，其中线程0进行迭代步0~49999，线程1进行迭代步50000~99999.
