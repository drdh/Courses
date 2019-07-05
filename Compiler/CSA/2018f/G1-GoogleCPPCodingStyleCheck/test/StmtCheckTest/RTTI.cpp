#include <typeinfo>
class X {
    public:
        X() {
            mX = 101;
        }
    private:
        int mX;
};

int main() {
    X x, *px;
	if(0){{{typeid(x).name(); }}}
	{{X *pxx = dynamic_cast<X*>(px);}}
    return 0;
};
