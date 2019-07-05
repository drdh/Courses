#pragma GCC diagnostic error "-std=c++11"  
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>



char filename[20]="hlm.txt";
char zipname[20]="hlm.zip";
char filename2[20]="hlm2.txt";

/********************结构体**********************/ 	

	
//文件读取
typedef struct{
	unsigned char ch1;
	long int weight1;
}InitFile; 
 
 typedef struct{
 	unsigned char ch2;
 	long int weight2;
    char hcode[256];
 }HelpFile;

//赫夫曼树 
typedef struct {
    unsigned char ch;
    long int weight;
    int parent,lchild,rchild;
}HTNode,*HuffmanTree;

//赫夫曼编码 
typedef char**  HuffmanCode;
 
/*******************全局变量************************/ 
HuffmanTree HT;
HuffmanCode HC;
 
/********************主要操作函数声明*********************************/ 
 void compress(); //压缩 
 void decompress();//解压缩 
 
/*******************次要操作函数声明*******************************************/

//构造赫夫曼树并求出编码 
void HuffmanCoding(InitFile *w,int n);
//从HT的1-n位中选出权值最小的两位，返回其下标为s1,s2 
void select(int i,int *s1,int *s2);



/*******************函数主体********************/

//1.压缩 
void compress(){
	char code[256][50];
    unsigned char savechar=0;
	long int ziplength=0;
    long int filelength=0;
	char inputfile[100],outputfile[100];
	FILE *fp,*cp;
	int count[256];
	int i;
	long int num=0;
	unsigned char ctemp;
/*	printf(">>>>filename:     ");
	scanf("%s",inputfile);
	printf(">>>>zip filename: ");
	scanf("%s",outputfile);
*/	//if((fp=fopen("E://test1.txt","rb"))==NULL) {printf("cannot open the inputfile");return;}
	//if((cp=fopen("E://result","wb"))==NULL) {printf("cannot open the outputfile");return;}
    if((fp=fopen(filename,"rb"))==NULL) {printf("cannot open the inputfile");return;}
	if((cp=fopen(zipname,"wb"))==NULL) {printf("cannot open the outputfile");return;}
	//统计文件 
	for(i=0;i<256;i++){
		count[i]=0;
	}
	
//	fseek(fp,-1L,2);
//	int wjcd=ftell(fp);
//	fseek(fp,0,0);
//	while((ctemp=fgetc(fp))!=EOF){
    fread(&ctemp,sizeof(unsigned char),1,fp);
    long int maxsize=pow(2,31)-10;
	while(!feof(fp)){
        filelength++;
        if(filelength>maxsize)
            break;
		count[ctemp]++;
//    while(wjcd!=ftell(fp)){
		fread(&ctemp,sizeof(unsigned char),1,fp);
//		printf("%c",ctemp);
		
	}
	
	InitFile store[257];
	store[num]={0,0};
	for(i=0;i<256;i++){
		if(count[i]!=0){
			store[num+1].ch1=i;
			store[num+1].weight1=count[i];
			num++;
		}	
	}

	
  //构造赫夫曼树并求出编码
  printf("-------------------------\n>>>>HuffmanCode\n-------------------------\n");
  HuffmanCoding(store,num); 
  
  
  // 写配置文件
  
//  HelpFile hp[num];
//  for(i=0;i<=num;i++){
//  	hp[i].ch2=store[i+1].ch1;
//  	hp[i].weight2=store[i+1].weight1;
//  	strcpy(hp[i].hcode,HC[i+1]); 
//  }
  
  FILE *ptr;
  if((ptr=fopen("helpbook","wb"))==NULL) {printf("cannot create the help file");return;}
  fwrite(&num,sizeof(int),1,ptr);
  fwrite(&num,sizeof(int),1,ptr);
//  fwrite(&filelength,sizeof(int),1,ptr);
  for(i=0;i<=num;i++){
  	fwrite(&store[i],sizeof(InitFile),1,ptr);
  }
  printf("---------------------\n>>>>helpbook has been writen\n");
  printf("num of node:%d\n",num);
  for(i=1;i<=num;i++){
  	printf("%c %d %s\n",i,i,HC[i]);
  }

   for(i=1;i<=num;i++){
    	strcpy(code[store[i].ch1],HC[i]); 
        printf("%x   %s   %s\n",store[i].ch1,code[store[i].ch1],HC[i]);
   }

  //储存
  	num=0;
  	fseek(fp,0,0);
//  	while((ctemp=fgetc(fp))!=EOF){
    
    
    
    char tempstr[256]={0};
    char write[9]={0};
    char copy[256]={0};
    fread(&ctemp,sizeof(unsigned char),1,fp);
    while(!feof(fp)){
        strcat(tempstr,code[ctemp]);
        while(strlen(tempstr)>=8)
        {
            strncpy(write,tempstr,8);
            write[8]='\0';
            strcpy(copy,&tempstr[8]);
            strcpy(tempstr,copy);
            
            savechar=0;
            for(i=0;i<=7;i++)
                savechar=(savechar<<1)+write[i]-'0';
            fwrite(&savechar,1,1,cp);
        }
        fread(&ctemp,sizeof(unsigned char),1,fp);
//while(1){
//		printf("大循环\n");
		 
/*  		for(i=0;i<strlen(code[ctemp]);i++){
//  			printf("小循环");
  			savechar|=code[ctemp][i]-'0';
  			num++;
  			if(num==8){
  				fwrite(&savechar,sizeof(unsigned char),1,cp);
  				ziplength++;
  				savechar=0;
  				num=0;
			}
			else{
				savechar=savechar<<1;
			}
			
		}
*/
		
	}
	if(strlen(tempstr))
    {
        strcpy(write,tempstr);
        int len=strlen(write);
        char temp[9]="00000000";
        int zero=8-len;
        strncat(write,temp,zero);
        savechar=0;
        for(i=0;i<=7;i++)
            savechar=(savechar<<1)+write[i]-'0';
        fwrite(&savechar,1,1,cp);
        rewind(ptr);
	    fwrite(&zero,sizeof(int),1,ptr);
    }
//	printf("结束循环\n");
/*	if(num!=8){
		savechar=savechar<<(8-num);
		fwrite(&savechar,sizeof(unsigned char),1,cp);
		ziplength++;
	}
*/
	printf("---------------------\n>>>>success\n----------------------\n");
//	printf("before:  %ld byte\n",filelength);
//	printf("after:   %ld byte\n",ziplength); 
//	printf("rate:    %.2f\n",1-(float)ziplength/filelength);
	free(HT);
	free(HC);
	fclose(ptr);
	fclose(fp);
	fclose(cp);
}
 
 
 //2.构造赫夫曼树并求出编码 
 void HuffmanCoding(InitFile *w,int n){
 	if(n<=1) return;
 	char *cd;
 	int m=2*n-1;
 	int j,k,f,start;
 	int s1,s2;
 	HuffmanTree p;
 	 
 	HT=(HuffmanTree)malloc((m+1)*sizeof(HTNode));
 	HT->ch=HT->weight=HT->lchild=HT->rchild=HT->parent=0; 
 	for(p=HT,j=0;j<=n;++j,++p,++w) {
 		p->ch=w->ch1;
 		p->weight=w->weight1;
 		p->lchild=p->rchild=p->parent=0;
	 }
 	for(;j<=m;++j,++p) {
 		p->ch=0;
 		p->weight=0;
 		p->lchild=p->rchild=p->parent=0;
	 }
		
 	for(j=n+1;j<=m;++j){
 		select(j-1,&s1,&s2);
 		HT[s1].parent=j;HT[s2].parent=j;
 		HT[j].lchild=s1;HT[j].rchild=s2;
 		HT[j].weight=HT[s1].weight+HT[s2].weight;
	 }
	    HT[m].parent=0;
	  
	 HC=(HuffmanCode)malloc((n+1)*sizeof(unsigned char*));
	 cd=(char *)malloc(n*sizeof(unsigned char));
	 cd[n-1]='\0';
	 
	 for(j=1;j<=n;++j){
	 	start=n-1;
	 	for(k=j,f=HT[j].parent;f!=0;k=f,f=HT[f].parent){
	 		if(HT[f].lchild==k) cd[--start]='0';
	 		else cd[--start]='1'; 
		 }
		 HC[j]=(char*)malloc((n-start)*sizeof(unsigned char));
		 strcpy(HC[j],&cd[start]);
		 printf("num of node:%3d HuffmanCode:%s\n",j,HC[j]);
	 }

	 free(cd);

 } 

//3.从HT的1-n位中选出权值最小且parent为0的两位，返回其下标为s1,s2 
void select(int n,int *s1,int *s2){
	int i=0,j=0,k=0,m=0;

	for(i=1;i<=n;i++){
		if(HT[i].parent==0) {m=i;break;}
	} 
	for(i=1,j=m;i<=n;i++){
		if((HT[i].weight<HT[j].weight)&&HT[i].parent==0) j=i; 
	}
	*s1=j;
	
	for(i=1;i<=n;i++){
		if((HT[i].parent==0)&&(i!=j)) {m=i;break;}
	} 
	
	for(i=1,k=m;i<=n;i++){
		if(i==j) continue;
		if((HT[i].weight<HT[k].weight)&&HT[i].parent==0) k=i; 
	}
	*s2=k;
}

//5.解压函数
int unzip(int z,int flag){
	if(flag==0) return HT[z].lchild;
	if(flag==1) return HT[z].rchild;
} 



//6.解压
void decompress(){
	FILE *fp,*cp,*hp;
	if((hp=fopen("helpbook","rb"))==NULL) {printf("cannot open the helpbook");return;}
	if((fp=fopen(zipname,"rb"))==NULL) {printf("cannot open the inputfile");return;}
	if((cp=fopen(filename2,"wb"))==NULL) {printf("cannot open the outputfile");return;}
	unsigned char reader,slidchar;
    unsigned char op;
 	int num=0,i=0,m=0,count=0,zero=0;
 	long int filelength=0;
 	//读配置文件 
    fread(&zero,sizeof(int),1,hp);
 	fread(&num,sizeof(int),1,hp);
// 	fread(&filelength,sizeof(long int),1,hp);
// 	printf("filelength=%d\n",filelength);
 	InitFile store[num+1];
 	m=2*num-1;
 	for(i=0;i<=num;i++){
  	fread(&store[i],sizeof(InitFile),1,hp);
    }
    printf("i=%d\n",i);
    //重构赫夫曼树
    printf("-------------------------\n>>>>HuffmanCode\n-------------------------\n");
    HuffmanCoding(store,num);
    
    
    int maxCode=0;
    for(i=1;i<=num;i++)
    {
        int len=strlen(HC[i]);
        if(len>maxCode)
            maxCode=len;
    }
    
   char tempstr[256]={0};
   char work[9]={0};
   char copy[256]={0};
   work[8]='\0';
   unsigned char c1,c2;
   fread(&c1,1,1,fp);
   fread(&c2,1,1,fp);
   while(!feof(fp))
   {
       int j;
       for(j=0;j<=7;j++,c1<<=1)
       {
           if(c1&0x80)
               work[j]='1';
           else
               work[j]='0';
       }
       strcat(tempstr,work);

       while(strlen(tempstr)>=maxCode)
       {
           int cur=m;
           int k=0;
           while(HT[cur].rchild)
           {
               if(tempstr[k]=='0')
                   cur=HT[cur].lchild;
               else
                   cur=HT[cur].rchild;
               k++;
           }
         //  unsigned char gg='1';
           unsigned char write=cur-1;
           fwrite(&HT[cur].ch,1,1,cp);

           strcpy(copy,&tempstr[k]);
           strcpy(tempstr,copy);
       }
       c1=c2;
       fread(&c2,1,1,fp);
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
   strcat(tempstr,work);
   while(strlen(tempstr))
   {
       int cur=m;
       int k=0;
       while(HT[cur].rchild)
       {
            if(tempstr[k]=='0')
                cur=HT[cur].lchild;
            else
                cur=HT[cur].rchild;
            k++;
        }
         //  unsigned char gg='1';
        unsigned char write=cur-1;
        fwrite(&HT[cur].ch,1,1,cp);

        strcpy(copy,&tempstr[k]);
        strcpy(tempstr,copy);
   }
    
    
    
    

	//解压缩 
/*	while(1){
		
//		printf("大循环\n"); 
		if(count==filelength) break;
		fread(&reader,sizeof(unsigned char),1,fp);
		printf("%d\n",reader);
		op=0x80;
		for(i=0;i<8;i++){
//			printf("小循环");
			slidchar=reader&op;
			reader=reader<<1;
			slidchar=slidchar>>7;
			z=unzip(z,slidchar-0);
			if(HT[z].lchild==0||HT[z].rchild==0){
				fwrite(&HT[z].ch,sizeof(unsigned char),1,cp);
				count++;
				z=2*num-1;
			}
		}
    }
*/
    printf("success\n");
 free(HT);
 free(HC);
 fclose(fp);
 fclose(hp);
 fclose(cp); 
} 


int main(){
    compress();
	decompress();
	return 0;
} 
 
 
