#ifndef _WORD_CHAIN_H_
#define _WORD_CHAIN_H_

#include<vector>
#include<string>
#include<map>

#define MAX_CELL 26
class Core{
    private:
        static void DFS(   int index, 
                    std::vector<int>&current_path,
                    char specific_tail,unsigned specific_num,
                    std::vector<std::pair<int,std::vector<int>>> &result,int depth);

        static std::vector<std::pair<int,std::vector<int>>> maxLength(std::vector<std::string>text,std::map<std::string,int>word2id,bool w_or_c,
                                char specific_head,char specific_tail,
                                unsigned specific_num);

    public:
        //如果传入 0，表示没有约束
        //假定单词都是小写的，且不重复，不做任何额外检查
        //head,tail=='\0'表示无约束
        static int gen_chain_word(char* words[], int len, char* *&result,char head, char tail);
        static int gen_chain_char(char* words[], int len, char* *&result,char head, char tail);
        static int gen_chain_num(char* words[], int len, char* **&result,char head, char tail,int num,int *&result_num);
        static int text_preprocess(std::string filename,char** &words);
};

#endif