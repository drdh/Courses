#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

char originf[]="origin.data";
char compressf[]="zip.data";
char uncompressf[]="finally.data";


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
   // if(s1==s1)
   //     printf("\n\n\n%d   %d\n\n\n",s1,Total);
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

/*    rewind(fo);
    int a[257];
    fread(a,sizeof(int),257,fo);
    int i;
    for(i=0;i<257;i++)
        printf("%d    %d\n",Weight[i-1],a[i]);
*/
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

/*    strcat(work,HuffmanCode[c]);
    printf("\n\n\n%s   %d\n\n\n",work,strlen(work));
*/
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

/*   int tt;
   for(tt=0;tt<=256;tt++)
       printf("%x  %d  \n",tt,Weight[tt]);
   printf("%d,%d\n",m,Total);
*/

//   printf("%d\n",Root[m].w);
    

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
         //  unsigned char gg='1';
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
         //  unsigned char gg='1';
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


int main(int argc,char *argv[])
{
    if(argc>=2)
    {
        strcpy(originf,argv[1]);
        if(argc>=3)
        {
            strcpy(compressf,argv[2]);
            if(argc>=4)
                strcpy(uncompressf,argv[3]);
        }
    }
    Compress();
    Uncompress();
/*    int i;
 //   Weigh();
//    for(i=0;i<256;i++)
//        printf("%x   %d\n",i,Weight[i]);
//    int total=HuffmanCoding();
    for(i=0;i<256;i++)
        printf("%x   %d   %s\n",i,Weight[i],HuffmanCode[i]);
    int j;
    for(i=0;i<255;i++)
        for(j=i+1;j<256;j++)
            if(Weight[i]&&Weight[j]&&!strcmp(HuffmanCode[i],HuffmanCode[j]))
                printf("\n\n\n%x  %x    %s\n\n\n",i,j,HuffmanCode[i]);
*/


/*    for(i=0;i<=256+total;i++)
        printf("%x\n",Root+i);
*/
}
