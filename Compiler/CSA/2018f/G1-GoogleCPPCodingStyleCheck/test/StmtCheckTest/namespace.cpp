#include <iostream>
#include <string>
#include <stdio.h>
#include <vector>

using namespace std;

namespace{
	void f(){
		int a = 1;
		a++;
	}
}

namespace{
	void fg(){
		int a = 2;
		a++;
	}
}


namespace test{
	void f(){
		int a = 3;
		a++;
	}
}


namespace std{
	void f(){
		int a = 3;
		a++;
	}
}

int main(){

}
