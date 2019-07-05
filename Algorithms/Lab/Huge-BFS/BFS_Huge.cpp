#include<cstdio>
#include<cstring>
#include<cstdlib>
#include<ctime>
#include<iostream>
#include<map>
#include<set>
#include<queue>
#include<deque>


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
    //FILE *file=fopen("/home/drdh/Downloads/temp/dataset/twitter_small.txt","r");
    FILE *file=fopen("/home/drdh/Downloads/temp/dataset/twitter_large.txt","r");
    char buffer[256];
    const char delim[]=", ";

	map<int,pair<vector<int>,bool>>graph;
    //map<int,pair<deque<int>,bool>>graph;
	
	clock_t start=clock();
	char *from,*to;
	long edge_count=0;
    while(fgets(buffer,255,file)){
        from=strtok(buffer,delim);
        to=strtok(NULL,delim);
		/*
		if(!graph[atoi(from)].first.find(atoi(to))){
			graph[atoi(from)].first.push_back(atoi(to));
        	graph[atoi(from)].second=false;
		}
		*/
		graph[atoi(from)].first.push_back(atoi(to));
        graph[atoi(from)].second=false;
		edge_count++;
    }
	
	cout<<"Adjacent Lists time: "<<(clock()-start)*1.0/CLOCKS_PER_SEC<<"s"<<endl;
	cout<<"Edge Total: "<<edge_count<<endl;

    
	start=clock();
	
	//set<int>visited;
	queue<int>todo;

	int src=3;
	//int src=14724347;
	todo.push(src);
	//visited.insert(src);
    graph[src].second=true;

	int count=0;

	while(todo.size()){
		auto item=todo.front();
		todo.pop();
		count++;
		//cout<<count<<" : "<<item<<endl;
        auto info=graph[item];
        
		for(auto next:info.first){
            if(!graph[next].second){
                todo.push(next);
                graph[next].second=true;
            }            
		}
	}
	cout<<"BFS time: "<<(clock()-start)*1.0/CLOCKS_PER_SEC<<"s"<<endl;
	cout<<"Total: "<<count<<endl;
    

    fclose(file);

	cout<<getValue()/1024.0<<"MB"<<endl;
}