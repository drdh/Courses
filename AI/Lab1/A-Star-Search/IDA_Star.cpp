#include<iostream>
#include<fstream>
#include<cstdlib>
#include<queue>
#include<vector>
#include<time.h>
#include<cstring>

using namespace std;

int ROW,COL,dst_row,dst_col;

int h[30][60];//h(n), heuristics, -1==wall
vector<char>Way;

bool IDA(int maxf,int depth,int i,int j){
    if(i==dst_row && j==dst_col){
        return true;
    }
    if(depth+h[i][j]> maxf){
        return false;
    }
    int pre_h;
    if(h[i][j+1]!=-1 ){
        Way.push_back('R');
        pre_h=h[i][j+1];
        h[i][j+1]=-1;
        if(IDA(maxf,depth+1,i,j+1)){
            return true;
        }
        h[i][j+1]=pre_h;
        Way.pop_back();
    }
    if(h[i+1][j]!=-1 ){
        Way.push_back('D');
        pre_h=h[i+1][j];
        h[i+1][j]=-1;
        if(IDA(maxf,depth+1,i+1,j)){
            return true;
        }
        h[i+1][j]=pre_h;
        Way.pop_back();
    }
    
    if(h[i-1][j]!=-1 ){
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

int main(int argc,char * argv[]){
    clock_t start_time,stop_time;
    string filename;
    if(!strcmp(argv[1],"-1")){
        filename="input1.txt";
        ROW=18;//30
        COL=25;//60
        dst_row=16;//28
        dst_col=24;//59
    }
    else if(!strcmp(argv[1],"-2")){
        filename="input2.txt";
        ROW=30;
        COL=60;
        dst_row=28;
        dst_col=59;
    }
    else{
        cout<<"arg error"<<endl;
        return 0;
    }
    
    //read file
    int h1[30][60];
    ifstream in(filename);
    for(int i=0;i<ROW;i++){
        char buffer[COL*3];
        in.getline(buffer,COL*3-2);
        for(int j=0;j<COL;j++){
            int tmp=buffer[j*2]-'0';
            if(tmp==1){
                h1[i][j]=-1;
            }
            else{
                h1[i][j]=abs(i-dst_row)+abs(j-dst_col);
            }
        }
    }//(1,0)==>(16,24)
    for(int i=0;i<ROW;i++){
        for(int j=0;j<COL;j++){
            h[i][j]=h1[i][j];
            if(h1[i][j]!=-1 && i-1>=0 && i+1<ROW && j-1>=0 && j+1<COL){
                int count=0;
                if(h1[i-1][j]!=-1)count++;
                if(h1[i+1][j]!=-1)count++;
                if(h1[i][j-1]!=-1)count++;
                if(h1[i][j+1]!=-1)count++;
                if(h1[i-1][j-1]!=-1)count++;
                if(h1[i-1][j+1]!=-1)count++;
                if(h1[i+1][j-1]!=-1)count++;
                if(h1[i+1][j+1]!=-1)count++;

                if(count==8)
                    h[i][j]=-1;
            }
            cout<<(h[i][j]==-1? 1: 0)<<" ";
        }
        cout<<endl;
    }

    start_time=clock();
    //cout<<h[1][0];
    for(int maxf=h[1][0];!IDA(maxf,0,1,0);maxf++)
        cout<<maxf<<endl;

    stop_time=clock();

    ofstream out("output_IDA.txt");
    out<<(double)(stop_time-start_time)/CLOCKS_PER_SEC<<endl;
    for(int i=0;i<Way.size();i++){
        out<<Way[i];
    }
    out<<endl<<Way.size()<<endl;
}