#include "syscall.h"

//向控制台输出字符串。
void printStr(char *str)
{
    int i = 0;
    while (str[i] != '\0')
        i++;
    Write(str, i, ConsoleOutput);
}

int getStrLength(char *str)
{
    int i = 0;
    while (str[i] != '\0')
    {
        i++;
        if (i > 128)
        {
            printStr(" Warning : Too long for a str!\n");
            break;
        }
    }
    return i;
}

int find(char *str, char ch)
{

    int count = 0;
    while (str[count] != '\0')
    {
        if (str[count] == ch)
        {
            return count;
            // WriteDigit(count);
        }
        count++;
    }
    return -1;
}

void zeroBuffer(char *buffer, int size)
{
    int i = 0;
    for (i = 0; i < size; i++)
    {
        buffer[i] = '\0';
    }
}

void trim(char *strIn, char *strOut)
{
    char *start, *end, *tmp;

    tmp = strIn;
    while (*tmp == ' ')
        ++tmp;
    start = tmp;

    tmp = strIn + getStrLength(strIn) - 1;
    while (*tmp == ' ')
        --tmp;
    end = tmp;

    for (tmp = start; tmp <= end; tmp++)
        *strOut++ = *tmp;
}

void getCmdsBySplit(char *buff, char cmdLine[][16], int *cmdNum)
{
    int i = 0, j = 0, cmdlength = 0;
    char tmp[16];

    *cmdNum = 0;
    for (i = 0; buff[i] != '\0'; i++)
    {
        j = i;
        for (cmdlength = 0; buff[j] != ';' && buff[j] != '\0'; j++, cmdlength++)
        {
            tmp[cmdlength] = buff[j];
        }
        tmp[cmdlength] = '\0';

        trim(tmp, cmdLine[*cmdNum]);
        zeroBuffer(tmp, 16);

        i = j;
        if (getStrLength(cmdLine[*cmdNum]) != 0)
            (*cmdNum)++;
    }
}

int main()
{
    OpenFileId input = ConsoleInput;   //控制台输入
    OpenFileId output = ConsoleOutput; //控制台输出
    int N = 8;                         //最多支持8条指令
    int M = 16;                        //每条指令长度最大为16
    char buff[N * M];                  //保存用户的输入
    char cmdLine[N][M];                //指令集合，每条指令最大长度M，最多N条
    int cmdNum;                        //指令数量
    int childID[N];                    //子进程数量，最多N个
    char *prompt = "\nnachos >> ";     //提示语
    int i;

    while (1)
    {
        printStr(prompt); //提示符

        zeroBuffer(buff, N * M);
        for (i = 0; i < N; i++)
            zeroBuffer(cmdLine[i], M);

        i = 0;
        do
        {
            Read(&(buff[i]), 1, input);
        } while (buff[i++] != '\n');
        buff[--i] = '\0'; //读来的命令
        if (i == 0)
            continue;
        getCmdsBySplit(buff, cmdLine, &cmdNum);

        { //begin:实验二的代码必须写在这里++++++++++++++++++++
            /* 
             * TODO:利用系统调用并行执行cmdLine里面的多条命令
            */
	    for(i=0;i<cmdNum;i++)
	    {
	      childID[i]=Fork();
	     /* if(child==0)
		Exec(cmdLine[i]);
	      else
		Join(child);
	      */
	      if(childID[i]==0)
		
	      //else
		Exec(cmdLine[i]);
		//else
		 // Join(childID[i]);
		  
	    }
	    for(i=0;i<cmdNum;i++)
	      Join(childID[i]);
	    //  Exec(cmdLine[i]);
        } //end: +++++++++++++++++++++++++++++++++++++++++

        for (i = 0; i < 100; i++)
            ;
    }
}