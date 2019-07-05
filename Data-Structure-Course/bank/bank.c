#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<unistd.h>

#define INTERTIME 6
#define MINTIME 3
#define MAXTIME 31
#define SLEEPTIME 0

//Event Type
typedef struct Event
{
    int time;
    int type;
}Event;

//EventList
typedef struct EventList
{
    Event data;
    struct EventList *next;
}EventNode,*Link;

//Event List Init
Link CreateLink()
{
    Link head=(Link)malloc(sizeof(EventNode));
    head->data.time=-1;
    head->data.type=-1;
    head->next=NULL;
    return head;
}

//Event List Insert
void Insert(Link head,int time,int type)
{
    Link new=(Link)malloc(sizeof(EventNode));
    new->data.time=time;
    new->data.type=type;
    Link prior=head;
    for(prior=head ; prior->next&& prior->next->data.time <=time;prior=prior->next)
        ;
    new->next=prior->next;
    prior->next=new;
}

Link GetFirst(Link head)
{
    Link temp=head->next;
    head->next=temp->next;
    return temp;
}


//Customer
typedef struct Customer
{
    int arrive;
    int during;
}Customer;

//Queue
typedef struct Queue
{
    Customer data;
    struct Queue * next;
}QueueNode,*Queue;

//Queue Init
Queue CreateQueue()
{
    Queue head=(Queue)malloc(sizeof(QueueNode));
    head->data.arrive=-1;
    head->data.during=-1;
    head->next=NULL;
    return head;
}

//InQueue
void InQueue(Queue head,int arrive,int during)
{
    Queue tail=head;
    while(tail->next)
        tail=tail->next;
    Queue new=(Queue)malloc(sizeof(QueueNode));
    new->data.arrive=arrive;
    new->data.during=during;
    new->next=NULL;
    tail->next=new;
}

//OutQueue
void OutQueue(Queue head)
{
    Queue first=head->next;
    head->next=first->next;
    if(first)
        free(first);
}

//MinQueue
int MinQueue(Queue * q)
{
    int count[5];
    int i=1;
    int min=0;
    int qu=1;
    Queue cur;
    for(i=1;i<=4;i++)
    {
        count[i]=0;
        cur=q[i]->next;
        while(cur)
        {
            count[i]++;
            cur=cur->next;

        }
    }
    for(min=count[1],i=1;i<=4;i++)
    {
        if(count[i]<min)
        {
            min=count[i];
            qu=i;
        }
    }
    return qu;
}

/*
void PrintL(Link head)
{
    Link cur=head->next;
    while(cur)
    {
        printf("{%d,%d}  ",cur->data.time,cur->data.type);
        cur=cur->next;
    }
    printf("\n\n");
}

void PrintQ(Queue head)
{
    Queue cur=head->next;
    while(cur)
    {
        printf("{%d,%d}  ",cur->data.arrive,cur->data.during);
        cur=cur->next;
    }
    printf("\n\n");
}
*/


Link ev; //event list
Event event; //a event,time,type
Queue q[5]; //1,2,3,4 queue
//Customer customer;  //a customer,arrive,during
int TotalTime,TotalNum;

void OpenForDay()
{
    TotalTime=0;
    TotalNum=0;
    ev=CreateLink();
    event.time=0;
    event.type=0;
    Insert(ev,0,0);
    int i;
    for(i=0;i<=4;i++)
        q[i]=CreateQueue();
}

void Arrive(int CloseTime)
{
    TotalNum++;

    //this arrival 
    //next arrival
   // srand((unsigned)time(NULL));
    int during=(rand()%(MAXTIME-MINTIME))+MINTIME;
    int inter=rand()%INTERTIME;
    int nextArrive=event.time+inter;
    if(nextArrive<CloseTime)
        Insert(ev,nextArrive,0);
    int min=MinQueue(q);
    InQueue(q[min],event.time,event.type);

    //the first one to leave
    if(!q[min]->next->next)
        Insert(ev,event.time+during,min);

   // printf("%d,%d\n",during,inter);

}

void Leave()
{
    int i=event.type;
    int arriveTime=q[i]->next->data.arrive;
    int duringTime=q[i]->next->data.during;
    TotalTime+=event.time-arriveTime;
    OutQueue(q[i]);


    if(q[i]->next)
    {
        Queue cur=q[i]->next;
        Insert(ev,event.time+cur->data.during,i);
    }
}

float Bank(int CloseTime)
{
    OpenForDay();
    while(ev->next)
    {
        Link temp=GetFirst(ev);
        event.time=temp->data.time;
        event.type=temp->data.type;
        free(temp);

        if(event.type==0)
            Arrive(CloseTime);
        else
            Leave();

    }

    free(ev);
    int i;
    for(i=1;i<=4;i++)
        free(q[i]);
    return (float)TotalTime/TotalNum;
}


int main()
{
    int i=0;
    for(i=720;i<=960;i++)
    {
       // sleep(SLEEPTIME);
        printf("%d -> %f\n",i,Bank(i));
    }
/*    Link head=CreateLink();
    Insert(head,5,1);
    PrintL(head);
    Insert(head,3,0);
    PrintL(head);
    Insert(head,4,2);
    PrintL(head);
    Delete(head);
    PrintL(head);


    int i=0;
    Queue q[5];
    for(i=1;i<=4;i++)
        q[i]=CreateQueue();
    InQueue(q[1],3,5);
    PrintQ(q[1]);
    InQueue(q[2],2,6);
    PrintQ(q[2]);
    OutQueue(q[2]);
    PrintQ(q[2]);
    Queue min=MinQueue(q);
    PrintQ(min);
*/
        
}
