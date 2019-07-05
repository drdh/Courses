//LAB2, simple shell
//support multiple program run in multi-process, separate cmd by ';'

#include "syscall.h"
#include "func.h"

#define MAX_CMDLINE_LENGTH 256   //max cmdline length in a line
#define MAX_CMD_LENGTH 16       //max single cmdline length
#define MAX_CMD_NUM 16          //max cmdline number in single cmdline

//parse cmdline, return cmd number
int parseCmd(char* cmdline, char cmds[MAX_CMD_NUM][MAX_CMD_LENGTH]){
    int i,j;
    int offset = 0;
    int cmd_num = 1;     //at least 1 cmd, then add 1 cmd per ';'
    char tmp[MAX_CMD_LENGTH];

    for(i=0;i<MAX_CMD_NUM&&offset<MAX_CMDLINE_LENGTH;i++){
        while(cmdline[offset]==' ') offset++;
        for(j=0;j<MAX_CMD_LENGTH-1;j++){
            if(cmdline[offset]==';'||cmdline[offset]=='\0'||cmdline[offset]=='\n') break;
            cmds[i][j] = cmdline[offset++];
        }
        cmds[i][j]='\0';
        if(cmdline[offset]=='\0'||cmdline[offset]=='\n') break;
        if(cmdline[offset]==';') {
            if(cmdline[offset+1]=='\n') return cmd_num;
            cmd_num++;
            offset++;
        }
    }
    
    return cmd_num;
}

void zeroBuff(char* buff, int size){
    int i;
    for(i=0;i<size;i++){
        buff[i]='\0';
    }
}

int
main()
{
    int cmd_num;
    int i,j;
    int pids[MAX_CMD_NUM];
    char cmdline[MAX_CMDLINE_LENGTH];
    char cmds[MAX_CMD_NUM][MAX_CMD_LENGTH];

    while(1){
        for(i=0;i<MAX_CMD_NUM;i++){
            zeroBuff(cmds[i],MAX_CMD_LENGTH);
        }
        printStr("nachos shell ->");
        Read(cmdline,MAX_CMDLINE_LENGTH,0);      //input cmdline

        cmd_num = parseCmd(cmdline, cmds);
        for(i=0;i<cmd_num;i++){
            pids[i] = Fork();
            if(pids[i]==0) {
                Exec(cmds[i]);
                return 0;
            }
        }
        for(i=0;i<cmd_num;i++) Join(pids[i]);
    }
}

