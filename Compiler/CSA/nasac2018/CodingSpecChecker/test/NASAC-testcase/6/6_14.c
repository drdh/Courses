// An example program to demonstrate working 
// of enum in C 
#include<stdio.h> 
  
enum week{Mon, Tue, Wed, Thur, Fri, Sat, Sun}; 
  
int main() 
{ 
    enum week day = Mon;  // bad: redundant initialization
    day = Wed; 
    printf("%d\n",day); 
    return 0; 
} 
