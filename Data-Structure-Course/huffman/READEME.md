## Huffman 算法压缩与解压
---
####  说明
1. 某技校数据结构课作业
2. 目前的算法只对有规律的文件有效。
> 此处的规律指的是0x00到0xff的分布不均匀
>
> txt文件不论汉字还是英文有效。jpg,pdf压缩无效。
3. 以下内容毫无逻辑，不具有参考价值。

---
#### 参考资料
- [c语言实现把‘0’和‘1’字符串转化为二进制压缩保存成二进制文件](http://blog.csdn.net/u013434915/article/details/41365073)
- [C语言文件操作详解](http://www.cnblogs.com/likebeta/archive/2012/06/16/2551780.html)
- [【小项目】用Huffman树实现文件压缩并解压](http://blog.csdn.net/pointer_y/article/details/53104339)
- [C语言读写两种方式ASII和二进制](http://www.cnblogs.com/wangzijing/archive/2013/03/02/2940466.html)
- [C语言位运算符：与、或、异或、取反、左移和右移](http://www.cnblogs.com/yezhenhan/archive/2011/11/06/2238452.html)
- [C语言文件操作函数](http://blog.csdn.net/u010994304/article/details/50265681)
- [Huffman压缩真正的C++实现](http://blog.csdn.net/small_hacker/article/details/52843738)
- [C语言基础 ASCII转换成二进制存入数组中](http://blog.csdn.net/yushaopu/article/details/51908962)
- [字符串化为二进制](https://zhidao.baidu.com/question/310042590.html)
- [C语言字符串函数大全](https://www.byvoid.com/zhs/blog/c-string)
- [基于Huffman编码的压缩解压程序](http://www.cnblogs.com/keke2014/p/3857335.html)
---
#### 思路建立
###### 1. 读写二进制文件
```
FILE *fi=fopen("origin.data","rb");
FILE *fo=fopen("zip.data","wb");

unsigned char c;
fread(&c,sizeof(unsigned char),1,fi);
fwrite(&c,sizeof(unsigned char),1,fo);

fclose(fi);
fclose(fo);
```
其他的文件操作，比如rewind，以及对数组、结构体读写略。
###### 2. 权重
```
int *Weight[256]={0};

  c : File
  Weight[c]++
```
仅对权重不为0的编码
###### 3. 编码后
```
char *HuffmanCode[256];256    //malloc分配各自的空间，每个编码的结尾为'\0'
HuffmanCode[c]    //c是 8 bits 可进行位操作
```
当前所得，密码本
```
HuffmanCode[c]="11110"    //位数不定，结尾是'\0'
```
分析
一个无符号数(0~255)，本身是
```
数 123
位 7b 01111011
字符 '{'
具体 printf("%d,%x,%c",123,123,123);
```
可能用到的字符串操作
```
strncpy(char *d,char *s,int n);
strcpy(char *d,char *s);

strncat()
strcat()

strchr(char *s,char c)
strstr(char *s1,char *s2)

strcmp(char *s1,char *s2)
strncmp()

strlen()
```

动态读取文件
```
使 unsigend char c=220   //或0xdc 或11011100
变成 unsigned char s[]="11011100"

    i:7~0
        c=(c<<1)+s[i]-'0'
```
将c写入文件
这之前先写入末尾补0个数(用一个int 占位)，以及权重(便于解码)

###### 4. 解压
读取权重，构建Huffman树
解码
```
unsigned char c=220    //11011100   0xdc
unsigned char str[9]
str[8]='\0'
   i:7~0
     str[i]=(c&0x80) ? '1':'0'
     c<<=1
```
解压用数逐位判断分支

压缩文件结构
```
int zero
0x00 int
...
0xff int
码
```

###### 5. 说明
位运算将c定义为unsigned char 否则解释为负
编码用前pow(2,31)-1个byte，或者考虑动态构建。

#### 代码说明
[代码位置](https://github.com/drdh/ADT/blob/master/huffman/hfm.c)
```
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

const char originf[]="origin.data";
const char compressf[]="zip.data";
const char uncompressf[]="finally.data";


int *Weight;
void Weigh()
{
    Weight=(int *)malloc(256*sizeof(int));
    int i;
    int allCount=0;
    int max=pow(2,31)-1;
    for(i=0;i<256;i++)
        Weight[i]=0;

    FILE * fp=fopen(originf,"rb");
    unsigned char c;
   // rewind(fp);
   fread(&c,1,1,fp);
    while(!feof(fp))
    {
       // fread(&c,1,1,fp);
        Weight[c]++;
        allCount++;
        fread(&c,1,1,fp);
        if(allCount>max-10)
            break;
    }
    
   // if(feof(fp))
   //     Weight[c]--;
    fclose(fp);
}

typedef struct{
    int w;
    int p,l,r;
}HTNode;

unsigned char **HuffmanCode;//code
HTNode *Root;      //tree
//int *Weight;

void SelectMin2(int Count,int Total,int *s1,int *s2)    //select 2 min weight , parent=0, weight !=0
{
    int min=Count+1;
    int i;
    *s1=789;
    *s2=789;
    for(i=1;i<=256+Total-2;i++)
    {
        if(Root[i].w &&  (!Root[i].p) &&  Root[i].w<min)
        {
            min=Root[i].w;
            *s1=i;
        }
    }
    
    Root[*s1].p=1;
    
    min=Count+1;
    for(i=1;i<=256+Total-2;i++)
    {
        if(Root[i].w && (!Root[i].p) && Root[i].w<min)
        {
            min=Root[i].w;
            *s2=i;
        }
    }
    Root[*s2].p=1;

}

int HuffmanCoding()
{
   // Weigh();
    int i;
    int Total=0;
    int Count=0;
    for(i=0;i<256;i++)
        if(Weight[i])
        {
            Total++;
            Count+=Weight[i];
        }
   // printf("%d\n",Count);
    int m=256+Total-1;   //total
    Root=(HTNode*)malloc((m+1)*sizeof(HTNode));    //tree

    //initial the tree
    
    for(i=1;i<=256;i++)
    {
        Root[i].w=Weight[i-1];
        Root[i].p=0;
        Root[i].l=0;
        Root[i].r=0;
    }

    for(;i<=m;i++)
    {
        Root[i].w=0;
        Root[i].p=0;
        Root[i].l=0;
        Root[i].r=0;
    }

    //build huffman tree
    for(i=257;i<=m;i++)
    {
        int s1,s2;
        SelectMin2(Count,Total,&s1,&s2);

        if(s1==s2)
            printf("\n\n\n%d   %d\n\n\n",i,s1);
        Root[s1].p=i;
        Root[s2].p=i;
        Root[i].l=s1;
        Root[i].r=s2;
        Root[i].w=Root[s1].w+Root[s2].w;
    }

    //get codes
    HuffmanCode=(unsigned char **)malloc((256)*sizeof(unsigned char*));
    unsigned char *temp=(unsigned char*)malloc((256*sizeof(unsigned char)));   //temp work place
    temp[256-1]='\0';   //end

    for(i=1;i<=256;i++)
    {
        if(!Root[i].w)    //weight ==0 --> p==0
        {
            HuffmanCode[i-1]=(unsigned char *)malloc(sizeof(unsigned char));
            *HuffmanCode[i-1]='\0';
        }
        else
        {
            int start=256-1;
            int c,f;
            for(c=i,f=Root[i].p;f!=0;c=f,f=Root[f].p)
            {
                if(Root[f].l==c)
                    temp[--start]='0';
                else temp[--start]='1';
            }
            HuffmanCode[i-1]=(unsigned char*)malloc((256-start)*sizeof(unsigned char));
            strcpy(HuffmanCode[i-1],&temp[start]);
        }
    }
    free(temp);
    return Total;
}

void Compress()
{
    Weigh();
    HuffmanCoding();
    FILE *fi=fopen(originf,"rb");
    FILE *fo=fopen(compressf,"wb+");

    int zero=0;
    fwrite(&zero,sizeof(int),1,fo);

    fwrite(Weight,sizeof(int),256,fo);

    unsigned char c;
    unsigned char w[9]={0};
    fread(&c,1,1,fi);
    unsigned char work[256]={0};
    unsigned char copy[256]={0};
    unsigned char write=0;
    while(!feof(fi))
    {
        strcat(work,HuffmanCode[c]);
        while(strlen(work)>=8)
        {
            strncpy(w,work,8);
            w[8]='\0';

            strcpy(copy,&work[8]);
            strcpy(work,copy);

            int i;
            write=0;
            for(i=0;i<=7;i++)
                write=(write<<1)+w[i]-'0';

            fwrite(&write,1,1,fo);
        }
        fread(&c,1,1,fi);

    }

    if(strlen(work))
    {
        strcpy(w,work);
        int len=strlen(w);
        unsigned char temp[9]="00000000";
        zero=8-len;
        strncat(w,temp,zero);
        int i;
        write=0;
        for(i=0;i<=7;i++)
            write=(write<<1)+w[i]-'0';
        fwrite(&write,1,1,fo);
        rewind(fo);
        fwrite(&zero,sizeof(int),1,fo);
    }

    fclose(fi);
    fclose(fo);

    int i;
    
        for(i=0;i<256;i++)
        printf("%x   %d   %s\n",i,Weight[i],HuffmanCode[i]);
        
    for(i=0;i<256;i++)
    {
        free(HuffmanCode[i]);
    }
    
    
    free(HuffmanCode);
    free(Weight);
    free(Root);
}

void Uncompress()
{
    FILE *fi=fopen(compressf,"rb");
    FILE *fo=fopen(uncompressf,"wb");

    int zero;
    fread(&zero,sizeof(int),1,fi);
    Weight=(int *)malloc(256*sizeof(int));
    fread(Weight,sizeof(int),256,fi);

   int  Total= HuffmanCoding();
   int  m=256+Total-1;

   int i;
   int maxCode=0;
   for(i=0;i<256;i++)
   {
       int len=strlen(HuffmanCode[i]);
       if(len>maxCode)
           maxCode=len;
   }

   unsigned char store[256]={0};
   unsigned char work[9]={0};
   unsigned char copy[256]={0};
   work[8]='\0';
   unsigned char c1,c2;
   fread(&c1,1,1,fi);
   fread(&c2,1,1,fi);
   while(!feof(fi))
   {
       int j;
       for(j=0;j<=7;j++,c1<<=1)
       {
           if(c1&0x80)
               work[j]='1';
           else
               work[j]='0';
       }
       strcat(store,work);

       while(strlen(store)>=maxCode)
       {
           int cur=m;
           int k=0;
           while(Root[cur].r)
           {
               if(store[k]=='0')
                   cur=Root[cur].l;
               else
                   cur=Root[cur].r;
               k++;
           }
           unsigned char write=cur-1;
           fwrite(&write,1,1,fo);

           strcpy(copy,&store[k]);
           strcpy(store,copy);
       }
       c1=c2;
       fread(&c2,1,1,fi);
   }



   work[8-zero]='\0';
   int j;
   for(j=0;j<=7-zero;j++,c1<<=1)
   {
       if(c1&0x80)
           work[j]='1';
       else
           work[j]='0';
   }
   strcat(store,work);
   while(strlen(store))
   {
       int cur=m;
       int k=0;
       while(Root[cur].r)
       {
            if(store[k]=='0')
                cur=Root[cur].l;
            else
                cur=Root[cur].r;
            k++;
        }
        unsigned char write=cur-1;
        fwrite(&write,1,1,fo);

        strcpy(copy,&store[k]);
        strcpy(store,copy);
   }
   
    for(i=0;i<256;i++)
    {
        free(HuffmanCode[i]);
    }
    free(HuffmanCode);
    free(Weight);
    free(Root);

    fclose(fi);
    fclose(fo);
}


int main()
{
    Compress();
    Uncompress();
}

```
---
#### 其他
[zip压缩算法详细分析及解压实例解释](http://www.cnblogs.com/esingchan/p/3958962.html)
