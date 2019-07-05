class A{
    virtual void doSomething() = 0;
    virtual void doSomething2() = 0;

};
class B{
    virtual void callSomeThing() = 0;
    virtual void callSomeThing2() = 0;
};
class C{
    void notPure();
};
class D:public A, public B, public C{
    void alsoNotPure();
};
int main(){
    return 0;
}