#include<omp.h>
#include<iostream>
#include<vector>
#include<algorithm>

#define THREADS_NUM 3

using namespace std;


void PSRS(vector<int>& array){
    vector<int>regular_sample;

    int length_per_part=array.size()/THREADS_NUM;

    omp_set_num_threads(THREADS_NUM);
    #pragma omp parallel
    {
        int id=omp_get_thread_num();//局部排序
        sort(array.begin()+id*length_per_part,(id+1)*length_per_part-1>array.size()?array.end():array.begin()+(id+1)*length_per_part-1);

        #pragma omp critical
        for(int k=0;k<THREADS_NUM;k++){//正则采样
            regular_sample.push_back(array[(id-1)*length_per_part+(k+1)*length_per_part/(THREADS_NUM+1)]);
        }
    }

    sort(regular_sample.begin(),regular_sample.end());//采样排序

    vector<int>pivot;
    for(int m=0;m<THREADS_NUM-1;m++){
        pivot.push_back(regular_sample[(m+1)*THREADS_NUM]);//选择主元
    }

    vector<int> pivot_array[THREADS_NUM][THREADS_NUM];
    #pragma omp parallel
    {
        int id=omp_get_thread_num();

        for(int k=0;k<length_per_part;k++){//全局交换
            for(int m=0;m<THREADS_NUM;m++){
                if(array[id*length_per_part+k]<pivot[m]){
                    pivot_array[id][m].push_back(array[id*length_per_part+k]);
                    break;
                }
                else if(m==THREADS_NUM-1){
                    pivot_array[id][m].push_back(array[id*length_per_part+k]);
                }
            }
        }
    }

    vector<int>array_per_thread[THREADS_NUM];
    #pragma omp parallel
    {
        int id=omp_get_thread_num();
        for(int k=0;k<THREADS_NUM;k++){
            for(auto item:pivot_array[k][id]){
                array_per_thread[id].push_back(item);
            }
        }
        sort(array_per_thread[id].begin(),array_per_thread[id].end());//局部排序
    }

    for(int i=0;i<THREADS_NUM;i++){
        for(auto item:array_per_thread[i]){
            cout<<item<<" ";
        }
    }
    cout<<endl;

}


int main(){
    vector<int> array={ 16,2,17,24,33,28,30,1,0,27,9,36,25, 37,34,23,19,18,11,7,21,13,8,35,12,29 , 6,3,4,14,22,15,32,10,26,31,20,5 };

    double begin,end;
    begin = omp_get_wtime();
    PSRS(array);
    end = omp_get_wtime();
    cout<<"The running time is "<<end-begin<<endl;
    return 0;
}