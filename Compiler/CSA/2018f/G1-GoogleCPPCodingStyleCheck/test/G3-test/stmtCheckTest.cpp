///////////////////////////////////////////////////////////////
/// Tested by G3 group
/// Found error!
/// Error details:
///     line 32: cl ++;              No warning is emitted when postfix increment appears in methods of a class
///     line 51: int a = NULL;       No warning is emitted, but 'NULL' should be used as pointer
///     line 83: myfunc(             No warning is emitted when lambda is used as arguments
////////////////////////////////////////////////////////////////

/// G1: Fixed!


#include <stdio.h>
#include <typeinfo>
#include <alloca.h>

class myclass {
    public:
        myclass() {
            num = 0;
        }
        myclass operator++(int){
            myclass temp = *this;
            ++ num;
            return temp;
        }
    private:
        int num;
};

class myclass2 {
    public:
        myclass2(myclass cl) {
            cl ++;
            num = 0;
        }
    private:
        int num;
};

int myfunc(int (*f)(int i)){
	return f(1);
}

int main(int argc, char *argv[])
{
    // RTTI
    myclass cl, *pcl;
    typeid(cl);
    dynamic_cast<myclass*>(pcl);

    // zero
    int a = NULL;
    char c = 0;
    int *p = 0;
    float f = 0;

    // ++
    for(int i = 0; i < 5; i ++) i --;
    a = c ++;
    cl ++;

    // exception
    try {
        a = 5;
    }
    catch (const char* msg) {
    }

    // cast
    pcl = (myclass *)p;

    // alloca
    alloca(sizeof(int));

    // lambda
    auto k = [a, c]() {
		return a + c;
	};

	auto l = [&](int i)->int {
        f = 0.0;
        for(int i = 0; i < 1; i ++){
            if(f < 1){
                f += 1;
                continue;
            }
        }
		return a + c;
	};

	myfunc(
        [](int i)->int {
			int a = 1;
			int b = 2;
            int ret = a + b;
            ret = ret * i;
            ret ++;
            return ret;
		}
    );

    return 0;
}
