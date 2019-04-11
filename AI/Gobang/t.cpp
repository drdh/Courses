#include<iostream>
#include<set>
#include<string>

using namespace std;

class Position{
    public:
        int x,y,score;
        Position(){}
        Position(int x,int y):x(x),y(y),score(0){}
        Position(int x,int y,int score):x(x),y(y),score(score){}
        bool operator <(const Position &pos) const {
            if (score != pos.score) {
                return score > pos.score;
            }
            if (x != pos.x) {
                return x < pos.x;
            }
            else {
                return y < pos.y;
            }
        }
};


int main(){
    string s="123";
    cout<<s.size()<<endl;
}
