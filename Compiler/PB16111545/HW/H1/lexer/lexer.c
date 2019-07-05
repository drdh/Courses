#include<stdio.h>
#include<string.h>

#define BUFSIZE 100
#define PATSIZE 20

int pos[PATSIZE][3];
int num=0;

void update_pos(int start,int end,int type)
{
    pos[num][0]=start;
    pos[num][1]=end;
    pos[num][2]=type;
    num++;
}

int main()
{
    char s[BUFSIZE];
    printf("Input strings :\n");
    scanf("%[^\n]",s);
    int i=0;
    int start=0;
    while(s[i]!='\0')
    {
        switch(s[i])
        {
            case '<':
                i++;
                switch(s[i])
                {
                    case '=':
                        update_pos(i-1,i,0);
                        break;
                    case '>':
                        update_pos(i-1,i,1);
                        break;
                    default:
                        update_pos(i-1,i-1,2);
                        i--;
                }
                break;
            case '=':
                update_pos(i,i,3);
                break;
            case '>':
                i++;
                switch(s[i])
                {
                    case '=':
                        update_pos(i-1,i,4);
                        break;
                    default:
                        update_pos(i-1,i-1,5);
                        i--;
                }
                break;
            case ' ':
                break;
            default:
                start=i;
                while(s[i+1]!='<'&&
                        s[i+1]!='='&&
                        s[i+1]!='>'&&
                        s[i+1]!=' '&&
                        s[i+1]!='\0')
                    i++;
                update_pos(start,i,6);

        }
        i++;
    }

    char tag[7][3]={"LE","NE","LT","EQ","GE","GT","OT"};
    int j;
    for(j=0;j<num;j++)
        printf("%d,%d,%s\n",pos[j][0],pos[j][1],tag[pos[j][2]]);
}
