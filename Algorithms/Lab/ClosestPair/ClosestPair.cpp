#include<algorithm>
#include<iostream>
#include<cmath>
#include<fstream>
#include<cstring>
#include<vector>

#define INFINITE 100000;

using namespace std;

struct Point{
    float x,y;
};

int compareX(const void *a,const void *b){
    Point *p1=(Point *)a;
    Point *p2=(Point *)b;
    return p1->x > p2->x ? 1 :-1;
}

int compareY(const void *a,const void *b){
    Point *p1=(Point *)a;
    Point *p2=(Point *)b;
    return p1->y > p2->y ? 1 :-1;
}

float distance(const Point p1,const Point p2){
    return sqrt((p1.x - p2.x)*(p1.x - p2.x)+
                ((p1.y - p2.y)*(p1.y - p2.y)));
}

float bruteForce(Point P[], int n,int *p1,int *p2) { 
    float min = INFINITE; 
    for (int i = 0; i < n; ++i) {
        for (int j = i+1; j < n; ++j) {
            if (distance(P[i], P[j]) < min){
                min = distance(P[i], P[j]); 
                *p1=i;
                *p2=j;
            }
        }
    }
    return min; 
}


float closest(Point P[], int n,int *p1,int *p2) { 
    if (n <= 3) 
        return bruteForce(P, n,p1,p2); 
  
    int mid = n/2; 
    Point midPoint = P[mid]; 
  
    int dl_p1,dl_p2,dr_p1,dr_p2;
    float dl = closest(P, mid,&dl_p1,&dl_p2); 
    float dr = closest(P + mid, n-mid,&dr_p1,&dr_p2); 
    
    float d;
    if(dl<dr){
        *p1=dl_p1;
        *p2=dl_p2;
        d=dl;
    }
    else{
        *p1=dr_p1+mid;
        *p2=dr_p2+mid;
        d=dr;
    }

    Point near[n]; 
    int map[n];
    int size = 0; 
    for (int i = 0; i < n; i++) 
        if (abs(P[i].x - midPoint.x) < d){
            near[size] = P[i];
            map[size]=i;
            size++; 
        } 
    int n1,n2;
    //qsort(near, size, sizeof(Point), compareY);  
    for (int i = 0; i < size; ++i) 
        //for (int j = i+1; j < size && (near[j].y - near[i].y) < d; ++j) 
        for (int j = i+1; j < size; ++j) 
            if (distance(near[i],near[j]) < d) {
                d = distance(near[i], near[j]); 
                *p1=map[i];
                *p2=map[j];
            }
    return d;
} 

float closestPair(Point P[], int n,int *p1,int *p2) { 
    qsort(P, n, sizeof(Point), compareX); 
    return closest(P, n,p1,p2);
} 
  
int main() { 
    vector<string> file={"data/test1.txt","data/test2.txt","data/test3.txt"};

    for(int i=0;i<=2;i++){
        Point P[20];
        int n=0;
        ifstream in(file[i]);
        char buffer[256];
        in.getline(buffer,256);
        in.getline(buffer,256);
        cout<<buffer<<endl;
        char *p=strtok(buffer,",");

        while(p){
            float x=atof(p);
            p=strtok(NULL,";");
            float y=atof(p);
            p=strtok(NULL,",");
            P[n].x=x;
            P[n].y=y;
            n++;
        }
        int c1,c2;
        printf("bruteForce result: %f ",bruteForce(P,n,&c1,&c2));
        printf("(%.2f,%.2f),(%.2f,%.2f)\n",P[c1].x, P[c1].y, P[c2].x, P[c2].y);
        printf("The smallest distance is %f ", closestPair(P, n,&c1,&c2)); 
        printf("(%.2f,%.2f),(%.2f,%.2f)\n\n",P[c1].x, P[c1].y, P[c2].x, P[c2].y);
    }
    return 0; 
}
