#include<iostream>  
#include<vector>
#include<ctime>
#include<cstdlib>

using namespace std;

#define N 99

//n 任务数; machineNum 机器数
int taskN=7,machineNum=3,minTime=200;  
int machineTime[8];  
//int t[19]={2,14,4,16,6,5,3};  
int schdule[19];
int bestSchdule[19];
int currentMaxTime=0;

//返回最大值，而不是位置
int findMax(int a[],int length){
    int max=0;
    for(int i=0;i<length;i++)
        if(a[i]>max){
            max=a[i];
        }
    return max;
}
//返回最小的位置
int findMin(int a[],int length){
    int min=66666;
    int pos;
    for(int i=0;i<length ;i++)
        if(a[i]<min){
            min=a[i];
            pos=i;
        }
    return pos;
}

int compare (const void * a, const void * b)
{
  return ( *(int*)b - *(int*)a );
}


void search(int task,int *machineTime,int *t,int available){  
    if(task==taskN) {  
        int tmp=findMax(machineTime,machineNum);  
        if(tmp<minTime) {
            minTime=tmp;
            for(int temp=0;temp<taskN;temp++)
                bestSchdule[temp]=schdule[temp];
        }   
        return;  
    }  
    for(int i=0; i<available; i++) {  //每一层都是一个任务的分配
        int next_available=available;
        if(i==available-1){
            next_available=(available<machineNum) ? available+1:machineNum;
        }
        machineTime[i]+=t[task];  //第task个任务选择
        schdule[task]=i;
        if(machineTime[i]<minTime)  
            search(task+1,machineTime,t,next_available);  
        machineTime[i]-=t[task];  
    }  
}  
int main(){  
    taskN=19;
    machineNum=8;
    int t[19]={39, 39, 23, 45, 100, 69, 21, 81, 39, 55, 20, 86, 34, 53, 58, 99, 36, 45, 46};

    //测试案例
    int NTest[3]={10,15,19};
    int KTest[3]={3,5,8};
    int TTest[3][19]={{47, 20, 28, 44, 21, 45, 30, 39, 28, 33},
                      {98, 84, 50, 23, 32, 99, 22, 76, 72, 61, 81, 39, 76, 54, 37},
                      {39, 39, 23, 45, 100, 69, 21, 81, 39, 55, 20, 86, 34, 53, 58, 99, 36, 45, 46}};
    
    for(int per_task=0;per_task<3;per_task++){
        taskN=NTest[per_task];
        machineNum=KTest[per_task];
        cout<<"*********Test: "<<per_task+1<<endl;
        for(int j=0;j<taskN;j++){
            t[j]=TTest[per_task][j];
            cout<<t[j]<<" ";
        }
        cout<<endl;
        

        clock_t start=clock();

        qsort(t,taskN,sizeof(int),compare);
        for (int i=0; i<machineNum; i++)  
            machineTime[i]=0; 

        for(int task=0;task<taskN;task++){
            int guess_best=findMin(machineTime,machineNum);
            machineTime[guess_best]+=t[task];  //第task个任务选择
            bestSchdule[task]=guess_best;
        }
        minTime=findMax(machineTime,machineNum);
        cout<<"suboptimal time: "<<minTime<<endl;
        for (int i=0; i<machineNum; i++)  
            machineTime[i]=0; 
        
        search(0,machineTime,t,1);  

        start=clock()-start;
        cout<<start*1.0/CLOCKS_PER_SEC<<"s"<<endl;

        cout<<"best time: "<<minTime<<endl; 
        for(int temp=0;temp<taskN;temp++){
            cout<<t[temp]<<": "<<bestSchdule[temp]<<"\t";
        }
        cout<<endl;
        for(int temp=0;temp<machineNum;temp++){
            cout<<"machine "<<temp<<": ";
            for(int ll=0;ll<taskN;ll++){
                if(bestSchdule[ll]==temp)
                    cout<<t[ll]<<" ";
            }
            cout<<endl;
        }
        cout<<"\n\n\n";

    }
    return 0;  
}  
