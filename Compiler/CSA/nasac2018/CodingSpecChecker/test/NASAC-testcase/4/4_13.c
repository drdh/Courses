// An example program to demonstrate working 
// of enum in C 
#include<stdio.h> 
  
enum week{Mon, Tue, Wed, Thur, Fri, liu, Sun};  // bad: Chinese Pinyin
  
int main() 
{ 
    enum week day; 
    day = Wed; 
    printf("%d\n",day); 
    return 0; 
} 
