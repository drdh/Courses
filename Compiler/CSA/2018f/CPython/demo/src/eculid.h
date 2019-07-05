#ifndef _eculid_h
#define _eculid_h

#include <stdio.h>

extern "C" namespace eculid {
    int gcd(int, int);

    class Net
    {
    public:
        Net();
        //~Net();
        void forward();
    };
}

#endif
