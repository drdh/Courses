#include<iostream>
#include<string>
#include<cstring>
#include<fstream>
#include <sstream>

#include "LongestWordChain.h"

using namespace std;

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
    string file_name;

    for(int i=0;i<argc;i++){
        if(strcmp(argv[i],"-w")==0){
            define_w=true;
        } 
        else if(strcmp(argv[i],"-c")==0){
            define_c=true;
        }
        else if(strcmp(argv[i],"-f")==0 && i+1<argc){
            file_name_pos=i+1;
            file_name=string(argv[file_name_pos]);
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
        ifstream t(file_name);
        if(!t){
            throw "File doesn't exist";
        }
        stringstream buffer;
        buffer << t.rdbuf();
        string text_original(buffer.str());

        string result=LWC(define_w,text_original,specific_num,specific_head,specific_tail);
        ofstream out("solution.txt");
        out<<result;
    }
    catch(const char *msg){
        cerr<<msg<<endl;
    }
}
