// An example program to demonstrate working 
// of enum in C 
#include<stdio.h> 
  
enum wk{Mon, Tue, Wed, Thur, Fri, liu, Sun};  // bad ET is abbreviation

int main() 
{ 
    enum wk day; 
    day = Wed; 
    printf("%d\n",day); 
    return 0; 
} 
