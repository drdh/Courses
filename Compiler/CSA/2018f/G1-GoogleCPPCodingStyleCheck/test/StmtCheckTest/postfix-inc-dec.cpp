#include <vector>
using namespace std;
int main() {
	int a = 1;
	a ++;
	++ a;
	for(int i = 0; i < 10; i ++);
}

void test(){
	int a = 1, *p = &a;
	int abc[1] = {a++};
	if(a++)p--;
	*p ++ = 3;
	while(a--)p--;
	switch(a++){
		case 1: p--; break;
		case 2: p--; break;
		default: p--; break;
	}
	label: a++;
	do a++;
	while(2);
}
