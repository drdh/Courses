#ifndef _LONGEST_WORD_CHAIN_H_
#define _LONGEST_WORD_CHAIN_H_

#include<vector>
#include<string>
#include<map>

#define MAX_CELL 26

using namespace std;

void DFS(int index, 
        vector<int>&current_path,
        char specific_tail,unsigned specific_num,
        vector<pair<int,vector<int>>> &result,int depth);

vector<pair<int,vector<int>>> maxLength(vector<string>text,map<string,int>word2id,bool w_or_c,
                                char specific_head,char specific_tail,
                                unsigned specific_num);

/*
    w_or_c -w -c 有且只有一个
    file_name 输入文件
    specific_num -n后的数字,为0则表示为指定
    specific_head -h后的字母，为'\0'则表示未指定
    specific_tail -t后的字母，为'\0'则表示未指定
    输出到solution.txt中
*/

string LWC(bool w_or_c,string text_original,unsigned specific_num,char specific_head,char specific_tail);

#endif
