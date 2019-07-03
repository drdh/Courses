#include "WordChain.h"
#include<iostream>

using namespace std;

int main(){
    char *words[]={"algebra","apple","zoo","elephant","under","fox","dog","moon","leaf","trick",
            "pseudopseudohypoparathyroidism"};//11
    
    char **result1;
    int len;

    len=Core::gen_chain_word(words,11,result1,'\0','\0');

    cout<<"***gen_chain_word***"<<endl;
    cout<<len<<endl;
    for(int i=0;i<len;i++)
        cout<<result1[i]<<endl;

    len=Core::gen_chain_char(words,11,result1,'\0','\0');
    cout<<"\n\n\n***gen_chain_char***"<<endl;
    cout<<len<<endl;
    for(int i=0;i<len;i++)
    {
        cout<<result1[i]<<endl;
    }
    

    char ***result2;
    int *result2_num;

    len=Core::gen_chain_num(words,11,result2,'\0','\0',2,result2_num);

    cout<<"\n\n\n***gen_chain_num***"<<endl;
    cout<<len<<endl;
    for(int i=0;i<len;i++){
        for(int j=0;j<result2_num[i];j++)
            cout<<result2[i][j]<<endl;
        cout<<endl;
    }

}