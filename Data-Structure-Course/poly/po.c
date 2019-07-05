#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define MAX_SIZE 100

typedef struct Term
{
	int exp;
	float coe;
}Term;

typedef struct Node
{
	Term data;
	struct Node *next;
}Node,*Link;

Link Copy(Link head);

Link Create(void)  //create an empty list with a head
{
	Link head=(Link)malloc(sizeof(Node));
	head->next=NULL;
	head->data.coe=0;
	head->data.exp=0;
	return head;
}

Link Tail(Link head)  //find a head's tail
{
	Link tail=head;
	while((tail->next)!=NULL)
	  tail=tail->next;
	return tail;
}

Link Prior(Link head,Link node)  //find a node's prior of a head,if it exits
{
	Link cur=head;  //current node when searching
	while(cur->next!=node)
	  cur=cur->next;
	return cur;
}

void Append(Link head,int exp,float coe)
{
	Link tail=Tail(head);
	Link newNode=(Link)malloc(sizeof(Node));
	newNode->data.exp=exp;
	newNode->data.coe=coe;
	newNode->next=NULL;
	tail->next=newNode;
}

Link FindMax(Link head)
{
	Link cur=head->next;  //cur
	int exp=0;
	Link max=head->next;
	while(cur)
	{
		if(cur->data.exp > exp)
		{
			exp=cur->data.exp;
			max=cur;
		}
		cur=cur->next;
	}
	return max;
}


Link Sort(Link head)    //sort
{
	Link newHead=Create();
	Link max;
	Link prior;
	Link temp;
	while(head->next)
	{
		max=FindMax(head);
		temp=newHead->next;
		newHead->next=max;
		prior=Prior(head,max);
		prior->next=max->next;
		max->next=temp;

	}
	free(head);
    Link p=newHead->next;
	Link n=newHead->next->next;

    while(n)
	{
		if(p->data.exp==n->data.exp)
		{
			p->data.coe+=n->data.coe;
			p->next=n->next;
			free(n);
			n=p->next;
		}
		else
		{
			p=p->next;
			n=n->next;
		}
	}

	p=newHead;
	n=newHead->next;

	while(n)
	{
		if(fabs(n->data.coe)<1e-6)
		{
			p->next=n->next;
			free(n);
			n=p->next;
		}
		else
		{
			p=p->next;
			n=n->next;
		}
	}
	return newHead;
}

Link Add(Link h1,Link h2)
{
	Link tail=Tail(h1);
    Link newHead;
    if(h2==h1)
      newHead=Copy(h2);
    else
      newHead=h2;
	tail->next=newHead;
    newHead->data.coe=0;
	newHead->data.exp=0;
//    if(h2!=h1)
//     	free(h2);
	h1=Sort(h1);
	return h1;
}


Link Subtract(Link h1,Link h2)
{
	Link cur=h2->next;
	while(cur)
	{
		cur->data.coe=-(cur->data.coe);
		cur=cur->next;
	}
	h1=Add(h1,h2);
	return h1;
}

Link Multiply(Link h1,Link h2)
{
    Link newHead=Create();
    Link n1;
    Link n2;
	if(h1==h2)
	  h2=Copy(h1);
    for(n1=h1->next;n1;n1=n1->next)
    {
        for(n2=h2->next;n2;n2=n2->next)
        {
            Append(newHead,n2->data.exp+n1->data.exp,(n2->data.coe)*(n1->data.coe));
        }
    }
    Link temp;
    for(n1=h1;n1;n1=temp)
    {
        temp=n1->next;
        free(n1);
    }
    for(n2=h2;n2;n2=temp)
    {
        temp=n2->next;
        free(n2);
    }
    newHead=Sort(newHead);
    return newHead;
}



void Print(Link head)
{
	Link cur=head->next;
	while(cur)
	{
	    printf("%.2fx^%d ",cur->data.coe,cur->data.exp);
		cur=cur->next;
	}
//    cur=cur->next;
//    printf("%.2fx^%d",cur->data.coe,cur->data.exp);
	printf("\n\n");
}



Link Copy(Link head)
{
    Link copy_head=Create();
    Link cur;

    for(cur=head->next; cur!=NULL ;cur=cur->next)
        Append(copy_head,cur->data.exp,cur->data.coe);
    return copy_head;
}

float Power(float x,int n)
{
    float result=1;
    while(n--)
      result*=x;
    return result;
}


float Cal(Link head,float x)
{
    float sum=0;
    Link cur;
    for(cur=head->next;cur;cur=cur->next)
      sum+=(cur->data.coe) * Power(x,cur->data.exp);
    return sum;
}

Link Input()
{
    printf("How many items ?\n");
    int count;
    scanf("%d",&count);
    Link head=Create();
    int exp;
    float coe;
    while(count--)
    {
        printf("exp=  ");
        scanf("%d",&exp);
        printf("coe=  ");
        scanf("%f",&coe);
        Append(head,exp,coe);
     }
    return head;
}

void Content()
{
    printf("-------------------------------\n");
    printf("|        New            n     |\n");
//    printf("    Sort           s     \n");
    printf("|        Calculate      c     |\n");
    printf("|        Print          p     |\n");
    printf("|        Add            +     |\n");
    printf("|        Subtract       -     |\n");
    printf("|        Multiply       *     |\n");
    printf("|        Quit           q     |\n");
    printf("-------------------------------\n\n");
}

void Print_all(Link *h,int input)
{
    
    if(input==0)
        printf("None    \n");
    int i=0;
    for(i=1;i<=input;i++)
    {
        printf("%d  ",i);
        if(!(h[i]->next))
        {printf("\n\n");continue;}
        h[i]=Sort(h[i]);
        Print(h[i]);
    }
}

int Check(int a,int b,int num)
{
    if(a<=0||b<=0||a>num||b>num)
    {
        printf("error\n\n");
        return 0;
    }
    else
        return 1;

}


int main()
{
    char choice;
    int num=0;
    Link h[MAX_SIZE];
    float x=0;
    int which=0;
    int a=0,b=0;
    Link tempa,tempb;

    Content();
    choice=getchar();
    getchar();
    while(choice!='q')
    {
        if(num>=99)
            break;
    switch(choice)
    {
        case 'n':
            printf("----new----\n");
            num++;
            h[num]=Input();
            getchar();
            Print(h[num]);
            break;

        case 'c':
            printf("----calculate----\n");
            Print_all(h,num);
            scanf("%d,%f",&which,&x);
            getchar();
            if(!Check(which,num,num))
                break;
            printf("%f\n",Cal(h[which],x));
            break;

        case 'p':
            printf("----Print----\n");
            Print_all(h,num);
            break;

        case '+':
            printf("----add----\n");
            scanf("%d,%d",&a,&b);
            getchar();
            if(!Check(a,b,num))
                break;
            tempa=Copy(h[a]);
            tempb=Copy(h[b]);
            num++;
            h[num]=Add(tempa,tempb);
            Print(h[num]);
            break;

        case '-':
            printf("----Subtract----\n");
            scanf("%d,%d",&a,&b);
            getchar();
            if(!Check(a,b,num))
                break;
            tempa=Copy(h[a]);
            tempb=Copy(h[b]);
            num++;
            h[num]=Subtract(tempa,tempb);
            Print(h[num]);
            break;
            
        case '*':
            printf("----multiply----\n");
            scanf("%d,%d",&a,&b);
            getchar();
            if(!Check(a,b,num))
                break;
            tempa=Copy(h[a]);
            tempb=Copy(h[b]);
            num++;
            h[num]=Multiply(tempa,tempb);
            Print(h[num]);
            break;
    }
    Content();
    choice=getchar();
    getchar();
    }
}
