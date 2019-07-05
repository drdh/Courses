#include "eculid.h"

namespace eculid {
    int gcd(int m, int n)
    {
        int i;
        if (m  < n)
            i = m;
        else
            i = n;
        for (; i > 0; i--)
            if (m % i == 0 && n % i == 0)
                break;
        return i;
    }

    Net::Net()
    {
        printf("This is constructor.\n");
        for(int i=0;i<10;i++){
            forward();
        }
    }

    /*
    Net::~Net()
    {
        printf("This is deconstructor.\n");
    }
    */

    void Net::forward()
    {
        printf("This is Net::forward\n");
    }
}
