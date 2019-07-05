#include<cstdio>
#include<cstring>
#include<cstdlib>
#include<ctime>
#include<iostream>
#include<map>
#include<set>
#include<queue>

using namespace std;

int parseLine(char* line){
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++;
    line[i-3] = '\0';
    i = atoi(p);
    return i;
}

int getValue(){ //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmSize:", 7) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}

int main(){
    FILE *file=fopen("/home/drdh/Downloads/temp/dataset/twitter_small.txt","r");
    //FILE *file=fopen("/home/drdh/Downloads/temp/dataset/twitter_large.txt","r");
    char buffer[256];
    const char delim[]=" \n,";

	map<int,set<int>>graph;
	
	clock_t start=clock();

	long edge_count=0;
    while(fgets(buffer,255,file)){
        char *from=strtok(buffer,delim);
        char *to=strtok(NULL,delim);
		graph[atoi(from)].insert(atoi(to));
		edge_count++;
    }
	
	cout<<"Adjacent Lists time: "<<(clock()-start)*1.0/CLOCKS_PER_SEC<<"s"<<endl;
	cout<<"Edge Total: "<<edge_count<<endl;
	start=clock();
	
	set<int>visited;
	queue<int>todo;

	int src=14724347;
	//int src=3;
	todo.push(src);
	visited.insert(src);

	int count=0;

	while(todo.size()){
		auto item=todo.front();
		todo.pop();
		count++;
		//cout<<count<<" : "<<item<<endl;

		for(auto next:graph[item]){
			if(!visited.count(next)){
				todo.push(next);
				visited.insert(next);
			}
		}
	}
	cout<<"BFS time: "<<(clock()-start)*1.0/CLOCKS_PER_SEC<<"s"<<endl;
	cout<<"Total: "<<count<<endl;
    fclose(file);

	cout<<getValue()/1024.0<<"MB"<<endl;
}