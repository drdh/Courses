#include<iostream>
#include<set>
#include<string>
#include<queue>
#include<cstring>

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

struct  cmp{
    bool operator()(Position &a,Position &b){
        return a.score<b.score;
    }
};

int scores[2][72];

int main(){
    memset(scores,0,sizeof(scores));
    for(int i=0;i<72;i++){
        cout<<scores[0][i]<<scores[1][i]<<endl;
    }
}
