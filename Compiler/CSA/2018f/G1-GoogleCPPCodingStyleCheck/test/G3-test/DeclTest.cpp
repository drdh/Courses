////////////////////////////////////////////////////////////////////////////////////////
///Tested by G3
///Test for DeclCheck
///Found error!
///Error message: 
///line 10: Variable-length argument lists should be considered when dealing with default arguments.
<<<<<<< HEAD
=======
///G1:
///BUG FIXED!
>>>>>>> 7361e5aea5bc61efef55fb823a2b993d4e722dc7
////////////////////////////////////////////////////////////////////////////////////////
#include <stdint.h>
#include <initializer_list>
#define mytype int
template<class... T> void test1(T... args){}
void test2(int &b, mytype c){}
inline int test3(){
	int i=0;
	if(i==0){
		for(;;);
	}
}
int main() {
	auto a={1,2};
	uint32_t b;
}
