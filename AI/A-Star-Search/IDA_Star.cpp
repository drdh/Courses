#include<iostream>
#include<fstream>
#include<cstdlib>
#include<queue>
#include<vector>
#include<time.h>

using namespace std;

int h[18][25];//h(n), heuristics, -1==wall
vector<char>Way;

bool IDA(int maxf,int depth,int i,int j){
    if(i==16 && j==24){
        return true;
    }
    if(depth+h[i][j]> maxf){
        return false;
    }
    int pre_h;
    if(h[i+1][j]!=-1){
        Way.push_back('D');
        pre_h=h[i+1][j];
        h[i+1][j]=-1;
        if(IDA(maxf,depth+1,i+1,j)){
            return true;
        }
        h[i+1][j]=pre_h;
        Way.pop_back();
    }
    if(h[i][j+1]!=-1){
        Way.push_back('R');
        pre_h=h[i][j+1];
        h[i][j+1]=-1;
        if(IDA(maxf,depth+1,i,j+1)){
            return true;
        }
        h[i][j+1]=pre_h;
        Way.pop_back();
    }
    if(h[i-1][j]!=-1){
        Way.push_back('U');
        pre_h=h[i-1][j];
        h[i-1][j]=-1;
        if(IDA(maxf,depth+1,i-1,j)){
            return true;
        }
        h[i-1][j]=pre_h;
        Way.pop_back();
    }
    if(h[i][j-1]!=-1){
        Way.push_back('L');
        pre_h=h[i][j-1];
        h[i][j-1]=-1;
        if(IDA(maxf,depth+1,i,j-1)){
            return true;
        }
        h[i][j-1]=pre_h;
        Way.pop_back();
    }
    return false;
}

int main(){
    clock_t start_time,stop_time;
    
    //read file
    ifstream in("input.txt");
    for(int i=0;i<18;i++){
        char buffer[64];
        in.getline(buffer,51);
        for(int j=0;j<25;j++){
            int tmp=buffer[j*2]-'0';
            if(tmp==1){
                h[i][j]=-1;
            }
            else{
                h[i][j]=abs(i-16)+abs(j-24);
            }
        }
    }//(1,0)==>(16,24)

    start_time=clock();
    for(int maxf=h[1][0];!IDA(maxf,0,1,0);maxf++);
    stop_time=clock();

    ofstream out("output_IDA.txt");
    out<<(double)(stop_time-start_time)/CLOCKS_PER_SEC<<endl;
    for(int i=0;i<Way.size();i++){
        out<<Way[i];
    }
    out<<endl<<Way.size()<<endl;
}