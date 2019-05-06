#include<iostream>
#include<fstream>
#include<cstdlib>
#include<queue>
#include<vector>
#include<time.h>
#include<cstring>

using namespace std;

int ROW,COL,dst_row,dst_col;

struct cmp{
    bool operator()(vector<int>&a,vector<int>&b){
        return a[2]+a[3]>b[2]+b[3];//<i,j,g,h>
    }
};

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

    int h[ROW][COL];//h(n), heuristics, -1==wall

    //read file
    ifstream in(filename);
    

    for(int i=0;i<ROW;i++){
        char buffer[COL*3];
        in.getline(buffer,COL*3-2);
        for(int j=0;j<COL;j++){
            int tmp=buffer[j*2]-'0';
            if(tmp==1){
                h[i][j]=-1;
            }
            else{
                h[i][j]=abs(i-dst_row)+abs(j-dst_col);
            }
        }
    }//(1,0)==>(16,24)

    //<i,j,g,h>
    priority_queue<vector<int>,vector<vector<int>>,cmp>Leaf;
    char Way[ROW][COL]; //L D R U

    start_time=clock();

    vector<int>pos={1,0,0,h[1][0]};//current position
    Leaf.push({1,0,0,h[1][0]});
    h[0][1]=-1; //never back

    while(!(pos[0]==dst_row && pos[1]==dst_col)){
        pos=Leaf.top();Leaf.pop();
        int i=pos[0],j=pos[1],g=pos[2];
        if(h[i+1][j]!=-1){
            Way[i+1][j]='D';
            Leaf.push({i+1,j,g+1,h[i+1][j]});
            h[i+1][j]=-1;
        }
        if(h[i][j+1]!=-1){
            Way[i][j+1]='R';
            Leaf.push({i,j+1,g+1,h[i][j+1]});
            h[i][j+1]=-1;
        }
        if(h[i-1][j]!=-1){
            Way[i-1][j]='U';
            Leaf.push({i-1,j,g+1,h[i-1][j]});
            h[i-1][j]=-1;
        }
        if(h[i][j-1]!=-1){
            Way[i][j-1]='L';
            Leaf.push({i,j-1,g+1,h[i][j-1]});
            h[i][j-1]=-1;
        }
    }

    stop_time=clock();
    ofstream out("output_A.txt");

    out<<(double)(stop_time-start_time)/CLOCKS_PER_SEC<<endl;
    int i=dst_row,j=dst_col;
    vector<char>tmp;
    while(!(i==1 && j==0)){
        switch (Way[i][j]){
            case 'D': i--;tmp.push_back('D'); break;
            case 'R': j--;tmp.push_back('R'); break;
            case 'U': i++;tmp.push_back('U'); break;    
            case 'L': j++;tmp.push_back('L'); break;
        }
    }
    for(int i=tmp.size()-1;i>=0;i--){
        out<<tmp[i];
    }
    out<<endl<<tmp.size()<<endl;
}