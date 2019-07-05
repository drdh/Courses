#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

#define infinity 47483647 
#define maxVertex 20
#define maxWord 15

typedef struct Graph{
    int arc[maxVertex][maxVertex];
    int verNum;
}Graph;

Graph *GraphInit(int verNum,int *init){    //interface,init:  weight,from,to
    Graph *G=(Graph*)malloc(sizeof(Graph));
    G->verNum=verNum;
    int i,j;
    for(i=0;i<verNum;i++){
        for(j=0;j<verNum;j++){
            G->arc[i][j]=infinity;
        }
        G->arc[i][i]=0;
    }
    for(i=0;init[i];i+=3){
        G->arc[init[i+1]][init[i+2]]=init[i];
    }
    return G;
}

int FindPath(Graph *G,int from,int to,int *path,int *distance){
    int final[maxVertex];
   // int distance[maxVertex];
    int i;
    for(i=0;i<G->verNum;i++){
        path[i]=-1;    //parent path for each node
        distance[i]=G->arc[from][i];
        final[i]=0;
        if(distance[i]<infinity){
            path[i]=from;    //parent path is from
        }
    }
    distance[from]=0;
    final[from]=1;
    for(i=0;i<G->verNum;i++){
        int min=infinity;
        int w;
        int cur=-1;
        for(w=0;w<G->verNum;w++){    //find currently shortest node
            if(!final[w]){
                if(distance[w]<min){
                    cur=w;
                    min=distance[w];
                }
            }
        }
        if(cur==-1)
            break;
       // int num=G->verNum;
        final[cur]=1;
       // printf("%d %d\n",i,cur);
      //  num=G->verNum;
        for(w=0; w<G->verNum;w++){
            if(!final[w] && (min+G->arc[cur][w] < distance[w])){
                distance[w]=min+G->arc[cur][w];
                path[w]=cur;    //parent node
            }
        }
    }
    free(G);
    return distance[to];
}

int Map(char (*word2int)[maxWord],char *word){
    char first;
    int i=0;
    first=word2int[i][0];
    while(first){
        if(!strcmp(word,word2int[i]))
            return i;
        i++;
        first=word2int[i][0];
    }
    strcpy(word2int[i],word);
    return i;
}

void Input(){
    int verNum;
    int init[3*maxVertex*maxVertex+1];
    int from;
    int to;
    printf("verNum:  ");
    scanf("%d",&verNum);
    int flag=0;
    printf("undigraph(0) or digraph(1)?  ");
    scanf("%d",&flag);

    char word2int[maxVertex][maxWord]={0};
    char word[2][maxWord];
    printf("Input departure,destination and distance\n");
    printf("terminate with $\n");
    int i=0;
    while(1){
       // char word[2][maxWord];
        int from,to;
        int weight;

        printf("departure:  ");
        scanf("%s",word[0]);
        if(word[0][0]=='$')
            break;

        printf("destination:  ");
        scanf("%s",word[1]);
        printf("distance:  ");
        scanf("%d",&weight);
        printf("\n");

        from=Map(word2int,word[0]);
        to=Map(word2int,word[1]);
        init[i]=weight;
        init[i+1]=from;
        init[i+2]=to;
        i=i+3;
    }
    if(flag)
        init[i]=0;
    else{
        int j=0;
        int stop=i;
        while(j<stop){
            init[i]=init[j];
            init[i+1]=init[j+2];
            init[i+2]=init[j+1];
            i+=3;
            j+=3;
        }
        init[i]=0;
    }
    printf("from:  ");
    scanf("%s",word[0]);
    printf("to:  ");
    scanf("%s",word[1]);

    from=Map(word2int,word[0]);
    to=Map(word2int,word[1]);
    
    int path[maxVertex];
    int distance[maxVertex];
    Graph *G=GraphInit(verNum,init);
    FindPath(G,from,to,path,distance);
    for(i=0;i<verNum;i++){
        printf("%s  %d\n",word2int[i],distance[i]);
        if(distance[i]==infinity){
            printf("unreachable\n\n");
            continue;
        }
        if(i==from)
            continue;
        int j=i;
        int record[maxVertex];
        int pos=0;
       // record[pos]=i;
        while(j!=from&&j>=0&&j<verNum){
            record[pos]=j;
            j=path[j];
           // printf("<--%s",word2int[j]);
            pos++;
           // record[pos]=j;
        }
        pos--;
        printf("%s-->%s",word2int[from],word2int[record[pos--]]);
        while(pos>=0)
            printf("-->%s",word2int[record[pos--]]);
        printf("\n\n");
    }
}


int main(){

Input();



    //printf("%d\n",infinity);
/*    int verNum=6;
    int init[91]={10,0,2,30,0,4,100,0,5,5,1,2,50,2,3,10,3,5,20,4,3,60,4,5,0};
    Graph *G=GraphInit(verNum,init);
    int path[6];
    int distance[6];
    FindPath(G,0,4,path,distance);
    int i;
    for(i=0;i<verNum;i++){
        printf("%d  %d\n",i,distance[i]);
        int j=i;
        while(j!=0&&j>=0&&j<verNum){
            j=path[j];
            printf("%d  ",j);
        }
        printf("\n\n");
    }
*/

}
