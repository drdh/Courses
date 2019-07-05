#include<stdio.h>
#include<stdlib.h>
int main()
{
    FILE *fp,*fo;
    fp=fopen("note","rb");
    fo=fopen("copy","wb");
    unsigned char c;

//复制
/*    fread(&c,1,1,fp);
    while(!feof(fp))
    {
        fwrite(&c,1,1,fo);
        fread(&c,1,1,fp);
        
    }

    rewind(fp);
*/
 
//读取验证
/*
    fread(&c,1,1,fp);
    
    int a[256]={0};
    printf("%d   %c   %d\n",c,c,a[c]);
    
*/

//字符串(8)以二进制解释为unsigned char
/*    unsigned char s[]="11011100";
    
    int i;
    c=0;
    for(i=0;s[i]!='\0';i++)
        c=(c<<1)+s[i]-'0';
    
    printf("%c   %x   %d\n",c,c,c);
    
*/

//unsigned char 以二进制转化为字符串(8)
/*    c=220;
    unsigned char str[9];
    
    int i;
    str[8]='\0';
    for(i=7;i>=0;i--)
    {
        str[i]=(c&1)?'1':'0';
        c=c>>1;
    }
    
    printf("%s\n",str);
*/
    
    putchar('\n');
    fclose(fp);
    fclose(fo);
}
