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

    vector<vector<int>>graph(11316811);
    vector<bool>visited(11316811,false);
	
	clock_t start=clock();
	char *from,*to;
	long edge_count=0;
    int int_from;
    long max=0;
    while(fgets(buffer,255,file)){
        from=strtok(buffer,delim);
        to=strtok(NULL,delim);
        graph[atoi(from)].push_back(atoi(to));
		edge_count++;
    }
	
	cout<<"Adjacent Lists time: "<<(clock()-start)*1.0/CLOCKS_PER_SEC<<"s"<<endl;
	cout<<"Edge Total: "<<edge_count<<endl;

    
	start=clock();
	
	queue<int>todo;
    int count=0;
    for(int i=0;i<11316811;i++){
        if(!visited[i]){
            int src=visited[i];
            todo.push(src);
            visited[src]=true;

            while(todo.size()){
                auto item=todo.front();
                todo.pop();
                count++;
                auto info=graph[item];
                
                for(auto next:info){
                    if(!visited[next]){
                        todo.push(next);
                        visited[next]=true;
                    }            
                }
            }
        }
    }	
	cout<<"BFS time: "<<(clock()-start)*1.0/CLOCKS_PER_SEC<<"s"<<endl;
	cout<<"Total: "<<count<<endl;

    fclose(file);
	cout<<getValue()/1024.0<<"MB"<<endl;
}