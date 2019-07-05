#include <initializer_list>
typedef int mytype;
struct test{
int ta;
char tb[5];
};
int a;
mytype c;
auto b=a;
auto d=c;
int main(){
int a;
mytype c;
auto b=a;
auto d=c;
auto e = {1.23,1.1};
auto f= {1};
auto g=3;
struct test test1={1,"12"};
auto test2=test1;
    return 0;
}
