#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include<sstream>
#include<cstring>

#define MAX_CELL 26

using namespace std;

vector<pair<int,string>> route[MAX_CELL][MAX_CELL];

void DFS(int index, 
        vector<string>&current_path,
        char specific_tail,unsigned specific_num,
        vector<pair<int,vector<string>>> &result,int depth){

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
            for(auto &word:route[index][i]){
                if(word.first!=0){//标0表示遍历过的边/word
                    int word_length=word.first;
                    word.first=0;
                    current_path.push_back(word.second);
                    
                    DFS(i,current_path,specific_tail,specific_num,result,depth+word_length);

                    word.first=word_length;
                    current_path.pop_back();
                }
            }
        }
    }
}

vector<pair<int,vector<string>>> maxLength(vector<string>text,bool w_or_c,
                                char specific_head,char specific_tail,
                                unsigned specific_num){
    int head[MAX_CELL]={0}; //head[i]有出边

    //填图
    for(auto node:text){
        head[node.front()-'a']=1;
        if(w_or_c){
            route[node.front()-'a'][node.back()-'a'].push_back(make_pair(1,node));//以word计数
        } 
        else{
            route[node.front()-'a'][node.back()-'a'].push_back(make_pair(node.size(),node));//以char 计数
        }
    }    
    
    //DFS搜索
    vector<pair<int,vector<string>>> result;
    result.push_back(make_pair<int,vector<string>>(0,{})); //没有-n则只有一个结果，保存在0;有-n则从1开始储存

    if(specific_head!='\0'){//指定了开头
        if(head[specific_head-'a']==1){
            vector<string>current_path={};
            DFS(specific_head-'a',current_path,specific_tail,specific_num,result,0); 
        }
    }
    else{
        for(int k=0;k<MAX_CELL;k++){//没有指定则遍历
            if(head[k]==1){
                vector<string>current_path={};
                DFS(k,current_path,specific_tail,specific_num,result,0); 
                cout<<k<<endl;
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

    try{
        if(file_name_pos==0){
            throw "No file name specialized";
        }
        if(!(define_c || define_w)){
            throw "-w,-c must be specialized";
        }
        if(define_c && define_w){
            throw "-w,-c cannot be specialized at the same time";
        }

        //读文件
        ifstream t(argv[file_name_pos]);
        if(!t){
            throw "File doesn't exist";
        }
    
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
        vector<pair<int,vector<string>>> result=maxLength(text,define_w,specific_head,specific_tail,specific_num);
        //输出
        ofstream out("solution.txt");
        if(specific_num!=0){
            out<<result.size()-1<<endl;
            for(int i=1;i<result.size();i++){
                for(auto word:result[i].second){
                    out<<word<<endl;
                }
                out<<endl;
            }
        }
        else{
            out<<result[0].first<<endl;
            for(auto word:result[0].second){
                out<<word<<endl;
            }
        }
    }
    catch(const char *msg){
        cerr<<msg<<endl;
    }
}