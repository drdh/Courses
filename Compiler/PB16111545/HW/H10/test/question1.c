#include<stdio.h>

typedef struct a{
    short d1_1;
    int d2;
    long d3;
    short d1_2;
}a;

typedef struct b{
    short d1;
    long d3;
    int d2;
}b;

typedef struct c{
    short d1_1,d1_2,d2_3,d1_4;
    long d3;
    int d2_1,d2_2;
}c;


int main(){
    printf("short=%d,int=%d,long=%d\n",sizeof(short),sizeof(int),sizeof(long));
    printf("a=%d,b=%d,c=%d\n",sizeof(a),sizeof(b),sizeof(c));
}