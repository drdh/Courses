#ifndef CLASS_DRAWER
#define CLASS_DRAWER

#include <map>
#include <vector>
#include <graphviz/gvc.h>

#include "Global_Data.h"
#include "Function.h"
#include "Call.h"

enum DRAWER {
    Graphviz, Print
};


class DrawerBase {
public:
    virtual void draw(Global_data data) = 0;
};

class DrawerGraphviz : public DrawerBase {
public:
    void draw(Global_data data);
};

class DrawerPrint : public DrawerBase {
public:
    void draw(Global_data data);
};

class DrawerPick {
public:
    DrawerBase *get(DRAWER type);
};

#endif