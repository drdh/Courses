class A{
public:
    A(){
        init();
    }
    virtual void init(){i = 0;}
    int i;
};
class B{
public:
    virtual void init(){ i = 0;}
    int i;
};
class C:public B{
public:
    C(){
        init();
    }
};