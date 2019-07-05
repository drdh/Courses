struct A{
    int x;
    int y;
    int z;
};
struct B{
    int x;
    int y;
    int getX()const{return x;}
    int setX(int _x){x = _x;}
};
struct C{
    int x;
    int y;
    void Initializer(){x = 0; y = 0;}
    void Reset(){x = 0; y = 0;}
};