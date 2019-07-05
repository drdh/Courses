#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<unistd.h>
//#include<Python.h>

int InterTime =10;
int MinTime= 3;
int MaxTime= 8;
int MinMoney=-3000;
int MaxMoney=3000;
int SleepTime= 0;
int MaxPeople=100;
int Total=2000;   //money
int Sleep=0;

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
    int arrive;    //arrive time for calculating the whole time
    int during;    //deal time
    int money;    // +  -
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
    head->data.money=0;
    head->next=NULL;
    return head;
}

//InQueue
void InQueue(Queue head,int arrive,int during,int money)
{
    Queue tail=head;
    while(tail->next)
        tail=tail->next;
    Queue new=(Queue)malloc(sizeof(QueueNode));
    new->data.arrive=arrive;
    new->data.during=during;
    new->data.money=money;
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

Queue PopQueue(Queue head)
{
    Queue first=head->next;
    head->next=first->next;
    return first;
}

Queue Search(Queue head)    //search and resort,return NULL if none
{
    Queue cur=head->next;
    Queue prior=head;
    while(cur)
    {
        if(cur->data.money+Total>=0)
        {
            prior->next=cur->next;
            cur->next=head->next;
            head->next=cur;
            return cur;
        }
        else
        {
            cur=cur->next;
            prior=prior->next;
        }

    }
    if(!cur)
        return NULL;
}



void PrintL(Link head)
{
    if(!head)
        return;
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
    if(!head)
        return;
    Queue cur=head->next;
    while(cur)
    {
        printf("{%d,%d,%d}  ",cur->data.arrive,cur->data.during,cur->data.money);
        cur=cur->next;
    }
    printf("\n\n");
}



Link ev; //event list
Event event; //a event,time,type,current event
Queue q[3]; //0//1,2 queue
//Customer customer;  //a customer,arrive,during
int TotalTime,TotalNum;
int TotalLeave=0;

void OpenForDay()
{
    TotalTime=0;
    TotalNum=0;
    ev=CreateLink();
    event.time=0;   //current evenr
    event.type=0;
    Insert(ev,0,0);   //insert the first arrive Event
    int i;
    for(i=0;i<=2;i++)
        q[i]=CreateQueue();
    
//output
    printf("Time= %d\n",event.time);
    sleep(Sleep);
    printf("open\n");
}

void Arrive(int CloseTime)
{
    TotalNum++;

    //this arrival 
    //next arrival
    //srand((unsigned)time(NULL));
    int during=(rand()%(MaxTime-MinTime))+MinTime;  //deal time
    int money= -(rand()%(MaxMoney-MinMoney))+MaxMoney;   //money
    int inter=rand()%InterTime;   //nextArrive
    int nextArrive=event.time+inter;  //current event time + intertime
    if(nextArrive<CloseTime&&(TotalNum-TotalLeave)<MaxPeople)
        Insert(ev,nextArrive,0);   //0 ,arrive event
    
    if(!q[1]->next)  //q[1] is empty
    {
        if(money+Total>=0)   //enable to deal
        {
            InQueue(q[1],event.time,during,money);
            Insert(ev,event.time+during,1);
        }
        else   //have to wait in q2
        {
                InQueue(q[2],event.time,during,money);  

        }
    }
    else  //q1 is not empty before
    {
        InQueue(q[1],event.time,during,money);
    }

            
    
/*    int min=MinQueue(q);
    InQueue(q[min],event.time,event.type);

    //the first one to leave
    if(!q[min]->next->next)
        Insert(ev,event.time+during,min);

   // printf("%d,%d\n",during,inter);
*/
}

void Leave(int CloseTime)
{

 /*   int inter=rand()%InterTime;   //nextArrive
    int nextArrive=event.time+inter;  //current event time + intertime
    if(nextArrive<CloseTime&&(TotalNum-TotalLeave)<MaxPeople)
        Insert(ev,nextArrive,0);   //0 ,arrive event
*/        
        
        
    int i=event.type;
    int arrive=q[i]->next->data.arrive;
    int during=q[i]->next->data.during;
    int money=q[i]->next->data.during;
    TotalTime+=event.time-arrive;
    Total+=money;
    OutQueue(q[i]);


    //search q2
    Queue temp=Search(q[2]);
    if(temp)  //find one
    {
        if(event.time+temp->data.during<=CloseTime)
            Insert(ev,event.time+temp->data.during,2);
        else
            PopQueue(q[2]);
    }

    if(i==1)
    {
        Queue cur=q[1]->next;
        Queue not;
        while(cur)
        {
            if(cur->data.money+Total>=0)  
            {
                Insert(ev,event.time+cur->data.during,1);
                break;
            }
            else
            {
                not=PopQueue(q[1]);
                InQueue(q[2],not->data.arrive,not->data.during,not->data.money);
                cur=q[1]->next;
                
            }

        }

    }

/*
    if(q[i]->next)
    {
        Queue cur=q[i]->next;
        Insert(ev,event.time+cur->data.during,i);
    }
*/
}

int a=0;
int b=0;
int c=0;

int Bank(int CloseTime)
{
    OpenForDay();
    PrintL(ev);
    PrintQ(q[1]);
    PrintQ(q[2]);
    while(ev->next)
    {
        PrintL(ev);
        PrintQ(q[1]);
        PrintQ(q[2]);
        printf("total=%d\n",Total);
        
        
        Link temp=GetFirst(ev);
        event.time=temp->data.time;
        event.type=temp->data.type;
        free(temp);

        printf("time=%d,type=%d\n",event.time,event.type);
        printf("arrive=%d,leave q1=%d,leave q2=%d\n",a,b,c);
       // PrintL(ev);

        if(event.time>CloseTime)
            break;

        if(event.type==0)
        {
            Arrive(CloseTime);
            a++;
        }
        //    Arrive(CloseTime);
        else
        {
            Leave(CloseTime);
            if(event.type==1)
            {    b++;
                TotalLeave++;
            }
            else
            {    c++;
                TotalLeave++;
            }
        }
           // Leave(CloseTime);

    }

    free(ev);
    int i;
    for(i=1;i<=2;i++)
        free(q[i]);


  //  PrintL(ev);
  //  PrintQ(q[1]);
  //  PrintQ(q[2]);

    return (int)TotalTime/TotalNum;
}


int main()
{
    //Bank(800);
    printf("%d",Bank(800));
    
}
