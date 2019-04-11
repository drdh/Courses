#include<iostream>
#include<cstdlib>
#include<set>
#include<vector>
#include<cmath>
#include<stack>
#include<string>
#include<algorithm>

#define BOARD_WIDTH 15
#define MAX_SCORE (10000000)
#define MIN_SCORE (-10000000)
int DEPTH = 7;

using namespace std;

class Position{
    public:
        int x,y,score;
        Position(){}
        Position(int x,int y):x(x),y(y),score(0){}
        Position(int x,int y,int score):x(x),y(y),score(score){}
        bool operator <(const Position &pos) const {
            if (score != pos.score) return score > pos.score;
            if (x != pos.x) return x < pos.x;
            else return y < pos.y;
        }
};

char board[BOARD_WIDTH][BOARD_WIDTH];
enum Role {EMPTY='0',HUMAN='1',AI='2'};
set<Position> currentPossiblePositions;

int winner;     //胜出者
//int scores[3][72];  //保存棋局分数（2个角色72行，包括横竖撇捺），0位置弃用
int allScore[2];    //局面总评分（2个角色）
Position searchResult;//存储搜索结果，即下一步棋子的位置



set<Position>oldPossiblePositions;
//当加上一个新的pos时，重新判断周围能下的位置(仅含周围相邻区域)
void AddPossiblePositions(const Position &pos){
    oldPossiblePositions=currentPossiblePositions;
    int poss_x,poss_y;
    vector<pair<int,int>>directions={{1,1},{1,-1},{-1,1},{-1,-1},{1,0},{0,1},{-1,0},{0,-1}};
    for(int i=0;i<directions.size();i++){
        poss_x=pos.x+directions[i].first;
        poss_y=pos.y+directions[i].second;
        //判断在棋盘内，而且没有被占有
        if(poss_x>=0 && poss_x<BOARD_WIDTH && poss_y>=0 && poss_y<BOARD_WIDTH && board[poss_x][poss_y]==EMPTY){
            currentPossiblePositions.insert(Position(poss_x,poss_y));
        }
    }
    if(currentPossiblePositions.find(pos)!=currentPossiblePositions.end()){
        currentPossiblePositions.erase(pos);//表示这个点已经被占据了
    }
}

void RollbackPossiblePositions(){
    currentPossiblePositions=oldPossiblePositions;
}

vector<pair<string,int>>paterns={
    {"11111",50000},
    {"011110",4320},
    {"011100",720},
    {"001110",720},
    {"011010",720},
    {"010110",720},
    {"11110",720},
    {"01111",720},
    {"11011",720},
    {"10111",720},
    {"11101",720},
    {"001100",120},
    {"001010",120},
    {"010100",120},
    {"000100",20},
    {"001000",20}
};

int evaluatePoint(const Position &p,char role){
    string lines[4];//横，竖，斜，反斜
    for (int i = max(0, p.x - 5); i <min(BOARD_WIDTH, p.x + 6); i++) {
        if (i != p.x) {
            lines[0].push_back(board[i][p.y] == role ? '1' : board[i][p.y] == EMPTY ? '0' : '2');
        }
        else {
            lines[0].push_back('1');
        }
    }
    for (int i = max(0, p.y - 5); i <min(BOARD_WIDTH, p.y + 6); i++) {
        if (i != p.y) {
            lines[1].push_back(board[p.x][i] == role ? '1' : board[p.x][i] == EMPTY ? '0' : '2');
        }
        else {
            lines[1].push_back('1');
        }
    }
    for (int i = p.x - min(min(p.x, p.y), 5), j = p.y - min(min(p.x, p.y), 5); i < min(BOARD_WIDTH, p.x + 6) && j <min(BOARD_WIDTH, p.y + 6); i++, j++) {
        if (i != p.x) {
            lines[2].push_back(board[i][j] == role ? '1' : board[i][j] == EMPTY ? '0' : '2');
        }
        else {
            lines[2].push_back('1');
        }
    }
    for (int i = p.x + min(min(p.y, BOARD_WIDTH - 1 - p.x), 5),j = p.y - min(min(p.y, BOARD_WIDTH - 1 - p.x), 5); i >= max(0, p.x - 5) && j < min(BOARD_WIDTH, p.y + 6); i--, j++) {
        if (i != p.x) {
            lines[3].push_back(board[i][j] == role ? '1' : board[i][j] == EMPTY ? '0' : '2');
        }
        else {
            lines[3].push_back('1');
        }
    }

    int result=0;
    for(int i=0;i<4;i++){
        for(int j=0;j<paterns.size();j++){
            int num=0;
            for(int t=0;(t=lines[i].find(paterns[j].first,t))!=string::npos; num++,t++);
            result+=num*paterns[j].second;
        }
    }
    return result;
}


