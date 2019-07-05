#ifndef _eculid_h
#define _eculid_h

#include <stdio.h>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <ctime>

extern "C" namespace eculid {
    int gcd(int, int);
    class Net
    {
    public:
        void forward();
        Net();
    };
}

#endif
