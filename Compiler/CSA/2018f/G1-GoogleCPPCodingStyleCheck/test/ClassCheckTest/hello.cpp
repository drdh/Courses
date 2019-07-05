class Hello{
public:
    Hello();
    ~Hello();
public:
    int pub_data;
    int get_pub_data()const{return pub_data;}
private:
    float private_data;

    float get_private_data()const;

};
int main(){
    return 0;
}
Hello::Hello():private_data(0){
    pub_data = 0;
}
Hello::~Hello(){

}
float Hello::get_private_data()const{
    return private_data;
}