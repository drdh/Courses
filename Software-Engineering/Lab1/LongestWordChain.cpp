#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include<sstream>
#include<algorithm>
#include<cstring>

#define MAX_CELL 26

using namespace std;

void DFS(int index,vector<int>route[MAX_CELL][MAX_CELL], 
        char specific_tail,unsigned specific_num,
        vector<pair<int,vector<int>>> &result,int depth){

    unsigned length=0;


}

vector<pair<int,vector<int>>> maxLength(vector<string>text,bool w_or_c,
                                char specific_head,char specific_tail,
                                unsigned specific_num){
    vector<int>route[MAX_CELL][MAX_CELL]; //构成26*26的多元邻接矩阵
    int head[MAX_CELL]={0}; //head[i]有出边

    //填图
    for(auto node:text){
        head[node.front()-'a']=1;
        if(w_or_c){
            route[node.front()-'a'][node.back()-'a'].push_back(1);//以word计数
        }
        else{
            route[node.front()-'a'][node.back()-'a'].push_back(node.size());//以char 计数
        }
    }    

    //DFS搜索
    vector<pair<int,vector<int>>> result;
    result.push_back(make_pair<int,vector<int>>(0,{})); //没有-n则只有一个结果，保存在0;有-n则从1开始储存

    if(specific_head!='\0'){//指定了开头
        if(head[specific_head-'a']==1){
            DFS(specific_head-'a',route,specific_tail,specific_num,result,0); 
        }
    }
    else{
        for(int k=0;k<MAX_CELL;k++){//没有指定则遍历
            if(head[k]==1){
                DFS(k,route,specific_tail,specific_num,result,0); 
            }
        }
    }

    return result;
}

int main(int argc,char *argv[]){
    //main参数处理的条件判断
    /*
        ./t -w -f ./text.txt 最多单词数量 4
        ./t -c -f ./text.txt 最多字母数量 4
        
        ./t -h e -w -f ./text.txt 指定开头字母 6
        ./t -t r -w -f ./text.txt 指定结尾字母 6
        ./t -h e -t r -c -f ./text.txt 指定开头和结尾字母 8

        ./t -n 4 -w -f ./text.txt 指定单词数量 6
        ./t -n 4 -c -f ./text.txt 指定字母数量 8

        ./t -h e -n 4 -w -f ./text.txt 指定开头字母且指定单词数量 8
        ./t -t r -w -n 4 -f ./text.txt 指定结尾字母指定单词数量 8
        ./t -h e -t r -n 4 -c -f ./text.txt 指定开头和结尾字母指定字母数量 10

    */
    bool define_w=false,define_c=false;//w,c
    unsigned file_name_pos=0,specific_num=0;//f,n
    char specific_head='\0',specific_tail='\0';//h,t

    for(int i=0;i<argc;i++){
        if(strcmp(argv[i],"-w")==0){
            define_w=true;
        } 
        else if(strcmp(argv[i],"-c")==0){
            define_c=true;
        }
        else if(strcmp(argv[i],"-f")==0 && i+1<argc){
            file_name_pos=i+1;
        }
        else if(strcmp(argv[i],"-n")==0 && i+1<argc){
            specific_num=atoi(argv[i+1]);
        }
        else if(strcmp(argv[i],"-h")==0 && i+1<argc){
            specific_head=argv[i+1][0];
        }
        else if(strcmp(argv[i],"-t")==0 && i+1<argc){
            specific_tail=argv[i+1][0];
        }
    } 
    if(file_name_pos==0){
        cerr<<"No file name specialized"<<endl;
    }
    if(!(define_c || define_w)){
        cerr<<"-w,-c must be specialized"<<endl;
    }
    if(define_c && define_w){
        cerr<<"-w,-c cannot be specialized at the same time"<<endl;
    }


    //读文件
    ifstream t(argv[file_name_pos]);
    
    stringstream buffer;
    buffer << t.rdbuf();
    string text_original(buffer.str());

    //text处理成vector<string>,仅包含小写字符
    vector<string>text;

    for(auto &c:text_original){
        if('a'<=c && c<= 'z')
            continue;
        else if('A'<=c && c<='Z'){
            c=c-'A'+'a';
            continue;
        }
        else
            c=' ';
    }
    size_t pos=0;
    string token;
    while((pos=text_original.find(' '))!=string::npos){
        token=text_original.substr(0,pos);
        if(token.length()!=0)
            text.push_back(token);
        text_original.erase(0,pos+1);
    }
    if(text_original.length()!=0)
        text.push_back(text_original);

    //计算
    vector<pair<int,vector<int>>> result=maxLength(text,define_w,specific_head,specific_tail,specific_num);



    
    



    //输出
}