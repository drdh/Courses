#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

#define _MAX_CELL 26

//DFS递归查询
pair<int, vector<int>> DFS(int index, vector<int> route[_MAX_CELL][_MAX_CELL])
{
    unsigned int length = 0;
    vector<int> rox;
    pair<int, vector<int>> rtn;
    rtn.second.push_back(index);

    //寻找下一跳
    for (int i = 0; i < _MAX_CELL; i++)
    {
        if (route[index][i].size() > 0)
        {
            sort(route[index][i].begin(), route[index][i].end());
            int l = route[index][i].back();
            route[index][i].pop_back();
            //进入下一深度之前需要把这条路径暂时删除，因为每个单词只能用一次
            pair<int, vector<int>> tx = DFS(i, route);
            unsigned int tmp = l + tx.first;
            route[index][i].push_back(l);
            if (length < tmp)
            {
                length = tmp;
                rox = tx.second;
            }
        }
    }
    rtn.first = length;
    rtn.second.insert(rtn.second.end(), rox.begin(), rox.end());
    return rtn;
}

void maxLength(vector<string> text)
{
    vector<int> route[_MAX_CELL][_MAX_CELL];
    //vector<int> path[_MAX_CELL][_MAX_CELL];
    int head[_MAX_CELL] = {0};

    //构建邻接图
    for (auto iter : text)
    {
        route[iter.front() - 'a'][iter.back() - 'a'].push_back(iter.size());
        head[iter.front() - 'a'] = 1;
    }

    //DFS遍历所有头节点
    pair<int, vector<int>> result = { 0, vector<int>{} };
    for (int k = 0; k < _MAX_CELL; k++)
        if (head[k] == 1)
        {
            pair<int, vector<int>> rt = DFS(k, route);
            if (rt.first > result.first)
            {
                result = rt;
            }
        }

    //打印
    for (int i = 0; i < result.second.size()-1; i++)
        for (auto iter = text.begin(); iter!= text.end(); iter++)
            if ((*iter).front() == ('a' + result.second[i]) && (*iter).back() == ('a' + result.second[i + 1]))
            {
                cout << (*iter) << endl;
                text.erase(iter);
                break;
            }

    return;
}

int main(int argc, char * argv[])
{
    //假设前级已经做了不合法数据验证
    maxLength(vector<string>{
        "apple",
        "elephant",
        "zoo",
        "under",
        "fox",
        "dog",
        "leaf",
        "tree",
        "text"
    });
/*
    maxLength(vector<string>{
        "apple",
        "blephant",
        "clephand",
        "flephant"
    });
*/
    return 0;
}