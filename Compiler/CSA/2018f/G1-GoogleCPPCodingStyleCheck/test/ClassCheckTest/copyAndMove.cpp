class A{
public:
    A(const A&);
    A(A &&);
public:
    A& operator=(const A&);
    A& operator=(const A&&);
};
class B{
public:
    B(B&);
};