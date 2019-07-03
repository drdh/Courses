#include <gtest/gtest.h>
#include "WordChain.h"

TEST(GenChainWordTest,NoHead_NoTail)
{
    char *words[]={"algebra","apple","zoo","elephant","under","fox","dog","moon","leaf","trick","ka"};
    char **result;
    char *realResult[5] = {"algebra","apple","elephant","trick","ka"};
    int len = Core::gen_chain_word(words,11,result,'\0','\0');
    ASSERT_EQ(len,5);
    for(int i = 0; i < 5; i++)
        EXPECT_STREQ(result[i],realResult[i]);
}

TEST(GenChainWordTest,Head_NoTail)
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

TEST(GenChainWordTest,NoHead_Tail)
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

TEST(GenChainWordTest,Head_Tail)
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

TEST(GenChainCharTest,NoHead_NoTail)
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

TEST(GenChainCharTest,NoHead_Tail)
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

TEST(GenChainCharTest,Head_NoTail)
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

TEST(GenChainCharTest,Head_Tail)
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

TEST(GenChainNumTest,Word_NoHead_NoTail)
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

TEST(GenChainNumTest,Word_Head_Tail)
{
    char *words[]={"algebra","apple","zoo","elephant","under","fox","dog","moon","leaf","trick",
            "pseudopseudohypoparathyroidism"};
    char ***result;
    int *len;
    int resultNum = Core::gen_chain_num(words,11,result,'a','t',3,len);
    char *realResult[1][3] = {{"algebra","apple","elephant"}};
    ASSERT_EQ(resultNum,1);
    for(int i = 0; i < 1; i++)
    {
        ASSERT_EQ(len[i],3);
        for(int j = 0; j < 3; j++)
            EXPECT_STREQ(result[i][j],realResult[i][j]);
    }
}

int main(int argc, char *argv[]) 
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}