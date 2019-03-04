#include<stdio.h>
#include<math.h>

float f(float x){
    return sqrtf(powf(x,2)+4)-2;
}

float g(float x){
    return powf(x,2)/(sqrtf(powf(x,2)+4)+2);
}

int main(){
    int i;
    float x;
    for(i=1;i<=10;i++){
        x=powf(8,-i);
        printf("x= %.12E\t",x);
        printf("g(8-%d) = %.12E \t",i,g(x));
        printf("f(8-%d) = %.12E \n",i,f(x));
    }
}