#include <gtest/gtest.h>
#include <string>
#include "WordChain.h"

TEST(GenChainWordTest,NoHead_NoTail)            //最多单词，不指定开头和结尾            
{
    char *words[]={"algebra","apple","zoo","elephant","under","fox","dog","moon","leaf","trick","Pseudopseudohypoparathyroidism"};
    char **result;
    char *realResult[4] = {"algebra","apple","elephant","trick"};
    int len = Core::gen_chain_word(words,11,result,'\0','\0');
    ASSERT_EQ(len,4);
    for(int i = 0; i < 4; i++)
        EXPECT_STREQ(result[i],realResult[i]);
}

TEST(GenChainWordTest,NoHead_NoTail_Circle1)    //最多单词，不指定开头和结尾，存在环
{
    char *words[]={"ac","abc","cab","ba"};
    char **result;
    char *realResult[4] = {"abc","cab","ba","ac"};
    int len = Core::gen_chain_word(words,4,result,'\0','\0');
    ASSERT_EQ(len,4);
    for(int i = 0; i < 4; i++)
        EXPECT_STREQ(result[i],realResult[i]);
}

TEST(GenChainWordTest,NoHead_NoTail_Circle2)    //最多单词，不指定开头和结尾，存在环    
{
    char *words[]={"aa","bb","cc","ab","bc","ca"};
    char **result;
    char *realResult[6] = {"aa","ab","bb","bc","cc","ca"};
    int len = Core::gen_chain_word(words,6,result,'\0','\0');
    ASSERT_EQ(len,6);
    for(int i = 0; i < 6; i++)
        EXPECT_STREQ(result[i],realResult[i]);
}

TEST(GenChainWordTest,Head_NoTail)              //最多单词，指定开头，不指定结尾
{
    char *words[]={"algebra","apple","zoo","elephant","under","fox","dog","moon","leaf","trick",
            "pseudopseudohypoparathyroidism"};
    char **result;
    char *realResult[2] = {"elephant","trick"};
    int len = Core::gen_chain_word(words,11,result,'e','\0');
    ASSERT_EQ(len,2);
    for(int i = 0; i < 2; i++)
        EXPECT_STREQ(result[i],realResult[i]);
}

TEST(GenChainWordTest,NoHead_Tail)              //最多单词，不指定开头，指定结尾
{
    char *words[]={"algebra","apple","zoo","elephant","under","fox","dog","moon","leaf","trick",
            "pseudopseudohypoparathyroidism"};
    char **result;
    char *realResult[3] = {"algebra","apple","elephant"};
    int len = Core::gen_chain_word(words,11,result,'\0','t');
    ASSERT_EQ(len,3);
    for(int i = 0; i < 3; i++)
        EXPECT_STREQ(result[i],realResult[i]);
}

TEST(GenChainWordTest,Head_Tail)                //最多单词，指定开头和结尾
{
    char *words[]={"algebra","apple","zoo","elephant","under","fox","dog","moon","leaf","trick",
            "pseudopseudohypoparathyroidism"};
    char **result;
    char *realResult[3] = {"algebra","apple","elephant"};
    int len = Core::gen_chain_word(words,11,result,'a','t');
    ASSERT_EQ(len,3);
    for(int i = 0; i < 3; i++)
        EXPECT_STREQ(result[i],realResult[i]);
}

TEST(GenChainCharTest,NoHead_NoTail)            //最多字母，不指定开头和结尾
{
    char *words[]={"algebra","apple","zoo","elephant","under","fox","dog","moon","leaf","trick",
            "pseudopseudohypoparathyroidism"};
    char **result;
    char *realResult[2] = {"pseudopseudohypoparathyroidism","moon"};
    int len = Core::gen_chain_char(words,11,result,'\0','\0');
    ASSERT_EQ(len,34);
    for(int i = 0; i < 2; i++)
        EXPECT_STREQ(result[i],realResult[i]);
}

TEST(GenChainCharTest,NoHead_Tail)              //最多字母，不指定开头，指定结尾
{
    char *words[]={"algebra","apple","zoo","elephant","under","fox","dog","moon","leaf","trick",
            "pseudopseudohypoparathyroidism"};
    char **result;
    char *realResult[2] = {"pseudopseudohypoparathyroidism","moon"};
    int len = Core::gen_chain_char(words,11,result,'\0','n');
    ASSERT_EQ(len,34);
    for(int i = 0; i < 2; i++)
        EXPECT_STREQ(result[i],realResult[i]);
}

TEST(GenChainCharTest,Head_NoTail)              //最多字母，指定开头，不指定结尾
{
    char *words[]={"algebra","apple","zoo","elephant","under","fox","dog","moon","leaf","trick",
            "pseudopseudohypoparathyroidism"};
    char **result;
    char *realResult[4] = {"algebra","apple","elephant","trick"};
    int len = Core::gen_chain_char(words,11,result,'a','\0');
    ASSERT_EQ(len,25);
    for(int i = 0; i < 4; i++)
        EXPECT_STREQ(result[i],realResult[i]);
}

TEST(GenChainCharTest,Head_Tail)                //最多字母，指定开头和结尾
{
    char *words[]={"algebra","apple","zoo","elephant","under","fox","dog","moon","leaf","trick",
            "pseudopseudohypoparathyroidism"};
    char **result;
    char *realResult[3] = {"algebra","apple","elephant"};
    int len = Core::gen_chain_char(words,11,result,'a','t');
    ASSERT_EQ(len,20);
    for(int i = 0; i < 2; i++)
        EXPECT_STREQ(result[i],realResult[i]);
}

TEST(GenChainNumTest,Word_NoHead_NoTail)        //指定长度的最多单词，不指定开头和结尾
{
    char *words[]={"algebra","apple","zoo","elephant","under","fox","dog","moon","leaf","trick",
            "pseudopseudohypoparathyroidism"};
    char ***result;
    int *len;
    int resultNum = Core::gen_chain_num(words,11,result,'\0','\0',3,len);
    char *realResult[2][3] = {{"algebra","apple","elephant"},{"apple","elephant","trick"}};
    ASSERT_EQ(resultNum,2);
    for(int i = 0; i < 2; i++)
    {
        ASSERT_EQ(len[i],3);
        for(int j = 0; j < 3; j++)
            EXPECT_STREQ(result[i][j],realResult[i][j]);
    }
}

TEST(GenChainNumTest,Word_Head_Tail)            //指定长度的最多单词，指定开头和结尾
{
    char *words[]={"algebra","apple","zoo","elephant","under","fox","dog","moon","leaf","trick",
            "pseudopseudohypoparathyroidism"};
    char ***result;
    int *len;
    int resultNum = Core::gen_chain_num(words,11,result,'a','t',3,len);
    char *realResult[2][3] = {{"algebra","apple","elephant"}};
    ASSERT_EQ(resultNum,1);
    for(int i = 0; i < 1; i++)
    {
        ASSERT_EQ(len[i],3);
        for(int j = 0; j < 3; j++)
            EXPECT_STREQ(result[i][j],realResult[i][j]);
    }
}

//以上为回归测试之前的功能是否正确

//以下测试新添加的接口text_preprocess是否正确
TEST(TextPreprocess,InvalidCharacter)           //测试函数text_preprocess，存在非法字符（非字母）的情形
{
	char **result;
	std::string filename = "./test/text_process1.txt";
        char *realResult[7] = {"abc","ac","acabca","ca","cb","cba","cbac"};
	int len = Core::text_preprocess(filename,result);
	ASSERT_EQ(len,7);
	for(int i = 0; i < 7; i++)
        EXPECT_STREQ(result[i],realResult[i]);
}

TEST(TextPreprocess,RepeatedCharacter)          //测试函数text_preprocess，存在重复单词的情形
{
	char **result;
	std::string filename = "./test/text_process2.txt";
	char *realResult[3] = {"ab","bc","ca"};
	int len = Core::text_preprocess(filename,result);
	ASSERT_EQ(len,3);
	for(int i = 0; i < 3; i++)
        EXPECT_STREQ(result[i],realResult[i]);
}

int main(int argc, char *argv[]) 
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}