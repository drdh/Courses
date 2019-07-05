#include<cstdio>
#include<cstring>
#include<iostream>

using namespace std;

#define MAX_LENGTH 100
char b[MAX_LENGTH][MAX_LENGTH];
char c[MAX_LENGTH][MAX_LENGTH];
int lcs;

void LCSLength(char*X, char*Y,int m,int n){
    for(int i=1;i<=m;i++)
        c[i][0]=0;
    for(int j=0;j<=n;j++)
        c[0][j]=0;

    for(int i=1;i<=m;i++){
        for(int j=1;j<=n;j++){
            if(X[i-1]==Y[j-1]){
                c[i][j]=c[i-1][j-1]+1;
                b[i][j]='=';
            }
            else if(c[i-1][j]>=c[i][j-1]){
                c[i][j]=c[i-1][j];
                b[i][j]='|';
            }
            else{
                c[i][j]=c[i][j-1];
                b[i][j]='-';
            }
        }
    }
}

void LCSPrint(char*X,int i,int j){
    if(i==0 || j==0)
        return;
    if(b[i][j]=='='){
        LCSPrint(X,i-1,j-1);
        cout<<X[i-1];
        lcs++;
    }
    else if(b[i][j]=='|')
        LCSPrint(X,i-1,j);
    else 
        LCSPrint(X,i,j-1);
}

int main(){
    char X[]="ABCBDAB";
    char Y[]="BDCABA";
    /*
    char X[100];
    char Y[100];
    cout<<"input X:"<<endl;
    cin>>X;
    cout<<"input Y:"<<endl;
    cin>>Y;
    */
    LCSLength(X,Y,strlen(X),strlen(Y));
    lcs=0;

    cout<<"term\t"<<"string\t"<<"length"<<endl;
    cout<<"X\t"<<X<<"\t"<<strlen(X)<<endl;
    cout<<"Y\t"<<Y<<"\t"<<strlen(Y)<<endl;

    cout<<"LCS\t";
    LCSPrint(X,strlen(X),strlen(Y));
    cout<<"\t"<<lcs<<endl;
}