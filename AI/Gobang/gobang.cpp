#include<iostream>
#include<cstdlib>
#include<set>
#include<vector>
#include<cmath>
#include<stack>
#include<string>
#include<algorithm>
#include<cstring>
#include<queue>
#include<ctime>

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
int allScore[2];    //局面总评分（2个角色）
Position searchResult;//存储搜索结果，即下一步棋子的位置


struct HistoryItem {
    set<Position> addedPositions;
    Position removedPosition;
};
vector<HistoryItem> history;
//当加上一个新的pos时，重新判断周围能下的位置(仅含周围相邻区域)
void AddPossiblePositions(const Position pos){
    int poss_x,poss_y;
    set<Position> addedPositions;
    vector<pair<int,int>>directions={{1,1},{1,-1},{-1,1},{-1,-1},{1,0},{0,1},{-1,0},{0,-1}};
    for(int i=0;i<directions.size();i++){
        poss_x=pos.x+directions[i].first;
        poss_y=pos.y+directions[i].second;
        //判断在棋盘内，而且没有被占有
        if(poss_x>=0 && poss_x<BOARD_WIDTH && poss_y>=0 && poss_y<BOARD_WIDTH && board[poss_x][poss_y]==EMPTY){
            pair<set<Position>::iterator, bool> insertResult = currentPossiblePositions.insert(Position(poss_x,poss_y));
            if(insertResult.second){
                addedPositions.insert(Position(poss_x,poss_y));
            }
        }
    }

    HistoryItem hi;
    hi.addedPositions = addedPositions;

    if(currentPossiblePositions.find(pos)!=currentPossiblePositions.end()){
        currentPossiblePositions.erase(pos);//表示这个点已经被占据了,需要从可放置集中删除
        hi.removedPosition = pos;
    }
    else{
        hi.removedPosition.x=-1;
    }
    history.push_back(hi);
}

void RollbackPossiblePositions(){
    if(currentPossiblePositions.empty()) return;

    HistoryItem hi=history.back();
    history.pop_back();
    for(auto i:hi.addedPositions){
        currentPossiblePositions.erase(i);
    }
    if(hi.removedPosition.x!=-1){
        currentPossiblePositions.insert(hi.removedPosition);
    }
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

int evaluatePoint(const Position p,char role){
    string lines[4];//横，竖，斜，反斜
    for(int i=max(0,p.x-5);i<min(BOARD_WIDTH,p.x+6);i++){
        if(i!=p.x){
            lines[0].push_back(board[i][p.y]==role? '1' : board[i][p.y] == EMPTY ? '0' : '2');
        }
        else{
            lines[0].push_back('1');
        }
    }
    for(int i=max(0,p.y-5);i<min(BOARD_WIDTH,p.y+6);i++){
        if(i!=p.y){
            lines[1].push_back(board[p.x][i]==role ? '1' : board[p.x][i] == EMPTY ? '0' : '2');
        }
        else{
            lines[1].push_back('1');
        }
    }
    for(int i=p.x-min(min(p.x, p.y),5), j=p.y-min(min(p.x, p.y),5); i<min(BOARD_WIDTH,p.x+6)&&j<min(BOARD_WIDTH,p.y+6);i++,j++) {
        if(i!=p.x){
            lines[2].push_back(board[i][j] == role ? '1' : board[i][j] == EMPTY ? '0' : '2');
        }
        else{
            lines[2].push_back('1');
        }
    }
    for(int i=p.x+min(min(p.y,BOARD_WIDTH-1-p.x),5),j=p.y-min(min(p.y,BOARD_WIDTH-1-p.x),5);i>=max(0,p.x-5)&&j<min(BOARD_WIDTH,p.y+6);i--,j++) {
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

int evaluate(Role role){
    return role==HUMAN ? allScore[0]:allScore[1];
}

int scores[2][72];  //保存棋局分数（2个角色72行，包括横竖撇捺）
void updateScore(const Position p){
    string lines1[4],lines2[4];//各个方向的状态
    int role=HUMAN;
    for(int i=0;i<BOARD_WIDTH;i++){//竖
        lines1[0].push_back(board[i][p.y] == role ? '1' : board[i][p.y] == EMPTY ? '0' : '2');
        lines2[0].push_back(board[i][p.y] == role ? '2' : board[i][p.y] == EMPTY ? '0' : '1');
    }
    for (int i = 0; i < BOARD_WIDTH; i++) { //横
        lines1[1].push_back(board[p.x][i] == role ? '1' : board[p.x][i] == EMPTY ? '0' : '2');
        lines2[1].push_back(board[p.x][i] == role ? '2' : board[p.x][i] == EMPTY ? '0' : '1');

    }
    for (int i = p.x - min(p.x, p.y), j = p.y - min(p.x, p.y); i < BOARD_WIDTH && j < BOARD_WIDTH; i++, j++) {//反斜杠
        lines1[2].push_back(board[i][j] == role ? '1' : board[i][j] == EMPTY ? '0' : '2');
        lines2[2].push_back(board[i][j] == role ? '2' : board[i][j] == EMPTY ? '0' : '1');

    }
    for (int i = p.x + min(p.y, BOARD_WIDTH - 1 - p.x), j = p.y - min(p.y, BOARD_WIDTH - 1 - p.x); i >= 0 && j < BOARD_WIDTH; i--, j++) {//斜杠
        lines1[3].push_back(board[i][j] == role ? '1' : board[i][j] == EMPTY ? '0' : '2');
        lines2[3].push_back(board[i][j] == role ? '2' : board[i][j] == EMPTY ? '0' : '1');
    }

    int line1Score[4],line2Score[4];//当前每个方向的得分
    memset(line1Score,0,sizeof(line1Score));
    memset(line2Score,0,sizeof(line2Score));

    for(int i=0;i<4;i++){
        for(int j=0;j<paterns.size();j++){
            int num=0;
            for(int t=0;(t=lines1[i].find(paterns[j].first,t))!=string::npos; num++,t++);
            line1Score[i]+=num*paterns[j].second;

            num=0;
            for(int t=0;(t=lines2[i].find(paterns[j].first,t))!=string::npos; num++,t++);
            line2Score[i]+=num*paterns[j].second;
        }
    }

    int a = p.y;
    int b = BOARD_WIDTH + p.x;
    int c = 2 * BOARD_WIDTH + (p.y - p.x + 10);
    int d = 2 * BOARD_WIDTH + 21 + (p.x + p.y - 4);

    //减去以前的记录
    for(int i = 0; i < 2; i++) {
        allScore[i] -= scores[i][a];
        allScore[i] -= scores[i][b];
    }

    //scores顺序 竖、横、\、/
    scores[0][a] = line1Score[0];
    scores[1][a] = line2Score[0];
    scores[0][b] = line1Score[1];
    scores[1][b] = line2Score[1];

    //加上新的记录
    for (int i = 0; i < 2; i++) {
        allScore[i] += scores[i][a];
        allScore[i] += scores[i][b];
    }

    if (p.y - p.x >= -10 && p.y - p.x <= 10) {
        for (int i = 0; i < 2; i++)
            allScore[i] -= scores[i][c];

        scores[0][c] = line1Score[2];
        scores[1][c] = line2Score[2];

        for(int i = 0; i < 2; i++)
            allScore[i] += scores[i][c];
    }

    if (p.x + p.y >= 4 && p.x + p.y <= 24) {
        for (int i = 0; i < 2; i++)
            allScore[i] -= scores[i][d];

        scores[0][d] = line1Score[3];
        scores[1][d] = line2Score[3];
        for (int i = 0; i < 2; i++)
            allScore[i] += scores[i][d];
    }
}

struct  cmp{
    bool operator()(Position &a,Position &b){
        return a.score<b.score;
    }
};


int abPruning(int depth,int alpha,int beta,Role currentSearchRole){
    int score1=evaluate(currentSearchRole);
    int score2=evaluate(currentSearchRole==HUMAN? AI:HUMAN);//component score

    if(score1>=50000){//当前局势已定胜负
        return MAX_SCORE - 1000 - (DEPTH - depth);
    }
    if(score2 >= 50000) {
        return MIN_SCORE + 1000 + (DEPTH - depth);
    }

    if(depth==0){//已到搜索深度
        return score1-score2;
    }

    priority_queue<Position,vector<Position>,cmp>possiblePositions;
    for(auto tmpPos:currentPossiblePositions){
        possiblePositions.push(Position(tmpPos.x,tmpPos.y,evaluatePoint(tmpPos,currentSearchRole)));//按照大根堆依次插入
    }

    int count=0;
    while(!possiblePositions.empty()){
        Position p=possiblePositions.top();
        possiblePositions.pop();

        board[p.x][p.y]=currentSearchRole;////放置棋子
        updateScore(p);

        p.score=0;
        AddPossiblePositions(p);//增加可能出现的位置

        int value=-abPruning(depth-1,-beta,-alpha,currentSearchRole==HUMAN?AI:HUMAN);

        RollbackPossiblePositions();//取消上一次增加的可能出现的位置
        board[p.x][p.y]=EMPTY;//取消放置
        updateScore(p);

        if(value>=beta){
            return beta;
        }
        if(value>alpha){
            alpha=value;
            if(depth==DEPTH){
                searchResult=p;
            }
        }

    /*    count++;
        if(count>=9){
            break;
        }
    */
    }
    return alpha;
}

//输出棋盘
void printBoard() {
    int i, j;
    cout<<"   ";
    for(i=0;i<BOARD_WIDTH;i++){
        if(i<10)
            cout<<"0 ";
        else
            cout<<"1 ";
    }
    cout<<"\n   ";
    for(i=0;i<BOARD_WIDTH;i++){
        cout<<i%10<<" ";
    }
    cout<<endl;
    for (i = 0; i < BOARD_WIDTH; i++) {
        if(i<10)
            cout<<'0'<<i<<" ";
        else
            cout<<i<<" ";

        for (j = 0; j < BOARD_WIDTH; j++) {
            cout<<(board[i][j]==EMPTY? '+' : board[i][j] )<< " ";
        }
        cout << endl;
    }
}

int main(){
    memset(board,EMPTY,BOARD_WIDTH*BOARD_WIDTH*sizeof(char));
    memset(scores,0,sizeof(scores));
    memset(allScore, 0, sizeof(allScore));
    currentPossiblePositions.clear();
    history.clear();
    winner=EMPTY;
    
    //开局
    srand((unsigned)time(NULL));
    //int x=rand()%BOARD_WIDTH;
    //int y=rand()%BOARD_WIDTH;
    int x=7,y=7;
    board[x][y]=AI;
    updateScore(Position(x,y));
    AddPossiblePositions(Position(x,y));

    //对弈
    while(winner==EMPTY){
        printBoard();
        while(board[x][y]!=EMPTY){
            cout<<"Next Move:";
            scanf("%d,%d",&x,&y);
        }
        board[x][y]=HUMAN;
        updateScore(Position(x,y));
        AddPossiblePositions(Position(x,y));
        
        int score=abPruning(DEPTH,MIN_SCORE,MAX_SCORE,AI);
        cout<<"AI:"<<searchResult.x<<","<<searchResult.y<<endl;
        if(score>=MAX_SCORE-1000-1){
            winner=AI;
        }
        else if(score<=MIN_SCORE+1000+1){
            winner=HUMAN;
        }
        board[searchResult.x][searchResult.y]=AI;
        updateScore(searchResult);
        AddPossiblePositions(searchResult);
    }
    printBoard();
    if(winner==HUMAN){
        cout<<"YOU WIN!"<<endl;
    }
    else{
        cout<<"AI WIN!"<<endl;
    }
}