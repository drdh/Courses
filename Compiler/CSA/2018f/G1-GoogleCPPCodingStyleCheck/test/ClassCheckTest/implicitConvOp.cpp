class A{
public:
    operator int(){return 5;}
    explicit operator float(){return 7.0;}
};
int main(){
    return 0;
}