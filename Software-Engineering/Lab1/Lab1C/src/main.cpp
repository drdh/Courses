#include "WordChain.h"
#include<iostream>
#include<cstring>
#include<fstream>

using namespace std;

int main(int argc,char *argv[]){
    //char *words[]={"algebra","apple","zoo","elephant","under","fox","dog","moon","leaf","trick",
    //        "pseudopseudohypoparathyroidism"};//11

    //main参数处理的条件判断
    /*
        ./t -w -f ./text.txt 最多单词数量 4
        ./t -c -f ./text.txt 最多字母数量 4
        
        ./t -h e -w -f ./text.txt 指定开头字母 6
        ./t -t r -w -f ./text.txt 指定结尾字母 6
        ./t -h e -t r -c -f ./text.txt 指定开头和结尾字母 8 

        ./t -n 4 -w -f ./text.txt 指定单词数量 6
        ./t -n 4 -c -f ./text.txt 指定字母数量 8 【不再支持】

        ./t -h e -n 4 -w -f ./text.txt 指定开头字母且指定单词数量 8
        ./t -t r -w -n 4 -f ./text.txt 指定结尾字母指定单词数量 8
        ./t -h e -t r -n 4 -c -f ./text.txt 指定开头和结尾字母指定字母数量 10 【不再支持】

    */
    bool define_w=false,define_c=false;//w,c
    unsigned file_name_pos=0,specific_num=0;//f,n
    char specific_head='\0',specific_tail='\0';//h,t
    string file_name;

    try{
        for(int i=0;i<argc;i++){
            if(strcmp(argv[i],"-w")==0){
                define_w=true;
            } 
            else if(strcmp(argv[i],"-c")==0){
                define_c=true;
            }
            else if(strcmp(argv[i],"-f")==0){
                if(i+1>=argc){
                    throw "-f with no specific args";
                }
                else{
                    file_name_pos=i+1;
                    file_name=string(argv[file_name_pos]);
                }
            }
            else if(strcmp(argv[i],"-n")==0){
                if (i+1>=argc) {
                    throw "-n with no specific args";
                }
                else {
                    specific_num=atoi(argv[i+1]);
                }
            }
            else if(strcmp(argv[i],"-h")==0){
                if (i+1>=argc) {
                    throw "-h with no specific args";
                }
                else {
                    specific_head=argv[i+1][0];
                }
            }
            else if(strcmp(argv[i],"-t")==0){
                if (i+1>=argc) {
                    throw "-t with no specific args";
                }
                else {
                    specific_tail=argv[i+1][0];
                }
            }
        } 

        if(file_name_pos==0 || file_name[0]=='-'){
            throw "No file name specialized";
        }
        if(!(define_c || define_w)){
            throw "-w,-c must be specialized";
        }
        if(define_c && define_w){
            throw "-w,-c cannot be specialized at the same time";
        }
        if(define_c && specific_num){
            throw "-c,-n cannot be specialized at the same time";
        }
        if(specific_head!='\0' && !(specific_head>='a' && specific_head <='z')){
            if(specific_head>='A' && specific_head <='Z')
                specific_head=specific_head-'A'+'a';
            else
                throw "-h char must be a-z or A-Z";
        }
        if(specific_tail!='\0' && !(specific_tail>='a' && specific_tail <= 'z')){
            if(specific_tail>='A' && specific_tail <= 'Z')
                specific_tail=specific_tail-'A'+'a';
            else
                throw "-t char must be a-z or A-Z";
        }


        char **words;
        int words_num=Core::text_preprocess(file_name,words);
        
        char **result1;
        int len;

        char ***result2;
        int *result2_num;

        ofstream out("solution.txt");

        if(define_w && !specific_num){
            len=Core::gen_chain_word(words,words_num,result1,specific_head,specific_tail);
            out<<len<<endl;
            for(int i=0;i<len;i++)
                out<<result1[i]<<endl;
        }
        else if(define_c && !specific_num ){
            len=Core::gen_chain_char(words,words_num,result1,specific_head,specific_tail);
            out<<len<<endl;
            for(int i=0;i<len;i++)
            {
                out<<result1[i]<<endl;
            }
        }
        else if(specific_num && define_w){
            len=Core::gen_chain_num(words,words_num,result2,specific_head,specific_tail,specific_num,result2_num);
            out<<len<<endl;
            for(int i=0;i<len;i++){
                for(int j=0;j<result2_num[i];j++)
                    out<<result2[i][j]<<endl;
                out<<endl;
            }
        }    
        else{
            throw "Args error";
        }
    }
    catch(const char *msg){
        cerr<<msg<<endl;
    }
}