#include "WordChain.h"

#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include<sstream>
#include<cstring>
#include<map>
#include<set>
#include<cstdlib>
#include<cstring>


using namespace std;
std::vector<std::pair<int,int>> route[MAX_CELL][MAX_CELL];

//假定单词都是小写的，且不重复
int Core::gen_chain_word(char* words[], int len, char* *&result,char head, char tail){
    vector<string>text;

    for(int i=0;i<len;i++){
        text.push_back(string(words[i]));
    }
    map<int,string>id2word; 
    map<string,int>word2id;
    int word_count=0;
    for(auto w:text){
            id2word.insert(make_pair(word_count,w));
            word2id.insert(make_pair(w,word_count));
            word_count++;            
    }

    if(word_count<2){
        throw "Too few words!";
    }

    //计算
    vector<pair<int,vector<int>>> result_max=maxLength(text,word2id,true,head,tail,0);
    
    //返回
    result=(char**)malloc(sizeof(char*)*result_max[0].first);

    vector<int>rr=result_max[0].second;
    for(int i=0;i<rr.size();i++){
        string word=id2word[rr[i]];
        result[i]=(char *)malloc(sizeof(char)*(word.size()+1));
        strcpy(result[i],word.c_str());
    }

    for(int i = 0; i < MAX_CELL; i++)
        for(int j = 0; j < MAX_CELL; j++)
            route[i][j].clear();

    return result_max[0].first;

}

int Core::gen_chain_char(char* words[], int len, char* *&result,char head, char tail){
    vector<string>text;

    for(int i=0;i<len;i++){
        text.push_back(string(words[i]));
    }
    map<int,string>id2word; 
    map<string,int>word2id;
    int word_count=0;
    for(auto w:text){
            id2word.insert(make_pair(word_count,w));
            word2id.insert(make_pair(w,word_count));
            word_count++;            
    }

    if(word_count<2){
        throw "Too few words!";
    }

    //计算
    vector<pair<int,vector<int>>> result_max=maxLength(text,word2id,false,head,tail,0);
    
    //返回
    result=(char**)malloc(sizeof(char*)*result_max[0].first);

    vector<int>rr=result_max[0].second;
    for(int i=0;i<rr.size();i++){
        string word=id2word[rr[i]];
        result[i]=(char *)malloc(sizeof(char)*(word.size()+1));
        strcpy(result[i],word.c_str());
    }

    for(int i = 0; i < MAX_CELL; i++)
        for(int j = 0; j < MAX_CELL; j++)
            route[i][j].clear();

    return result_max[0].first;   
}

int Core::gen_chain_num(char* words[], int len, char* **&result,char head, char tail,int num,int *&result_num){
    vector<string>text;

    for(int i=0;i<len;i++){
        text.push_back(string(words[i]));
    }
    map<int,string>id2word; 
    map<string,int>word2id;
    int word_count=0;
    for(auto w:text){
            id2word.insert(make_pair(word_count,w));
            word2id.insert(make_pair(w,word_count));
            word_count++;            
    }

    if(word_count<2){
        throw "Too few words!";
    }

    //计算
    vector<pair<int,vector<int>>> result_max=maxLength(text,word2id,true,head,tail,num);
    
    //返回
 
    result=(char***)malloc(sizeof(char**)*(result_max.size() - 1));
    result_num=(int *)malloc(sizeof(int)*(result_max.size() - 1));

    for(int j=1;j<result_max.size(); j++){
        vector<int>rr=result_max[j].second;
        result[j-1]=(char **)malloc(sizeof(char*)*rr.size());
        result_num[j-1]=rr.size();
        for(int i=0;i<rr.size();i++){
            string word=id2word[rr[i]];
            result[j-1][i]=(char *)malloc(sizeof(char)*(word.size()+1));
            strcpy(result[j-1][i],word.c_str());
        }
    }

    for(int i = 0; i < MAX_CELL; i++)
        for(int j = 0; j < MAX_CELL; j++)
            route[i][j].clear();

    return result_max.size()-1;
}


void Core::DFS(int index, 
        vector<int>&current_path,
        char specific_tail,unsigned specific_num,
        vector<pair<int,vector<int>>> &result,int depth){

    //判断是否达到要求
    if(specific_num!=0){//指定了-n,从1开始储存
        if(depth==specific_num){
            if(specific_tail!='\0' && specific_tail-'a' != index){//指定了tail, 但是不符合
                return;
            }
            else if(current_path.size()>1){//符合要求, 单词链至少两个单词
                result.push_back(make_pair(depth,current_path));
                return;
            }
        }
        else if(depth > specific_num){
            return;
        }
    }
    else{
        if(depth>result[0].first && !(specific_tail!='\0' && specific_tail-'a' != index) && current_path.size()>1){//储存在0
            result[0]=make_pair(depth,current_path);
        }
    }

    //寻找下一个节点
    for(int i=0;i<MAX_CELL;i++){
        if(route[index][i].size()>0){
            auto word=route[index][i].back();
            route[index][i].pop_back();
            current_path.push_back(word.second);

            DFS(i,current_path,specific_tail,specific_num,result,depth+word.first);

            route[index][i].push_back(word);
            current_path.pop_back();
        }
    }
}

vector<pair<int,vector<int>>> Core::maxLength(vector<string>text,map<string,int>word2id,bool w_or_c,
                                char specific_head,char specific_tail,
                                unsigned specific_num){
    int head[MAX_CELL]={0}; //head[i]有出边

    //填图
    for(auto node:text){
        head[node.front()-'a']=1;
        if(w_or_c){
            route[node.front()-'a'][node.back()-'a'].push_back(make_pair(1,word2id[node]));//以word计数
        } 
        else{
            route[node.front()-'a'][node.back()-'a'].push_back(make_pair(node.size(),word2id[node]));//以char 计数
        }
    }    
    
    //DFS搜索
    vector<pair<int,vector<int>>> result;
    result.push_back(make_pair<int,vector<int>>(0,{})); //没有-n则只有一个结果，保存在0;有-n则从1开始储存

    if(specific_head!='\0'){//指定了开头
        if(head[specific_head-'a']==1){
            vector<int>current_path={};
            DFS(specific_head-'a',current_path,specific_tail,specific_num,result,0); 
        }
    }
    else{
        for(int k=0;k<MAX_CELL;k++){//没有指定则遍历
            if(head[k]==1){
                vector<int>current_path={};
                DFS(k,current_path,specific_tail,specific_num,result,0); 
            }
        }
    }

    return result;
}
