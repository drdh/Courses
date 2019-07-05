// scheduler.cc
//	Routines to choose the next thread to run, and to dispatch to
//	that thread.
//
// 	These routines assume that interrupts are already disabled.
//	If interrupts are disabled, we can assume mutual exclusion
//	(since we are on a uniprocessor).
//
// 	NOTE: We can't use Locks to provide mutual exclusion here, since
// 	if we needed to wait for a lock, and the lock was busy, we would
//	end up calling FindNextToRun(), and that would put us in an
//	infinite loop.
//
// 	Very simple implementation -- no priorities, straight FIFO.
//	Might need to be improved in later assignments.
//
// Copyright (c) 1992-1996 The Regents of the University of California.
// All rights reserved.  See copyright.h for copyright notice and limitation
// of liability and disclaimer of warranty provisions.

#include "copyright.h"
#include "debug.h"
#include "scheduler.h"
#include "main.h"

static char statusStr[][20] = {
    "JUST_CREATED",
    "RUNNING",
    "READY",
    "BLOCKED"};

int threadCompareByPriority(Thread *x, Thread *y)
{
    if (x->getPriority() > y->getPriority())
        return -1;
    else if (x->getPriority() == y->getPriority())
        return 0;
    else
        return 1;
}

//----------------------------------------------------------------------
// Scheduler::Scheduler
// 	Initialize the list of ready but not running threads.
//	Initially, no ready threads.
//----------------------------------------------------------------------

Scheduler::Scheduler()
{
    readyList = new SortedList<Thread *>(threadCompareByPriority);
    toBeDestroyed = NULL;

    lastSwitchTick = kernel->stats->totalTicks;
}

//----------------------------------------------------------------------
// Scheduler::~Scheduler
// 	De-allocate the list of ready threads.
//----------------------------------------------------------------------

Scheduler::~Scheduler()
{
    delete readyList;
}

//----------------------------------------------------------------------
// Scheduler::ReadyToRun
// 	Mark a thread as ready, but not running.
//	Put it on the ready list, for later scheduling onto the CPU.
//
//	"thread" is the thread to be put on the ready list.
//----------------------------------------------------------------------

void Scheduler::ReadyToRun(Thread *thread)
{
    ASSERT(kernel->interrupt->getLevel() == IntOff);
    DEBUG(dbgThread, "Putting thread on ready list: " << thread->getName());

    thread->setStatus(READY);
    readyList->Insert(thread);
}

// begin: TODO:该函数的作用是调整所有就绪线程的优先级++++++++++++++++++++
void Scheduler::flushPriority() //
{
 	ListIterator<Thread *> *iter=new ListIterator<Thread *>(readyList);
	//iter=ListIterator<Thread *>(readyList);
	for (; !iter->IsDone(); iter->Next()) {
	  Thread *t=iter->Item();  
	  t->setPriority(t->getPriority()+AdaptPace);
	}
	Thread *t=kernel->currentThread;
	int priority=t->getPriority();
	priority=priority-(kernel->stats->totalTicks-lastSwitchTick)/100;
	t->setPriority(priority);

}
//end: ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//----------------------------------------------------------------------
// Scheduler::FindNextToRun
// 	Return the next thread to be scheduled onto the CPU.
//	If there are no ready threads, return NULL.
// Side effect:
//	Thread is removed from the ready list.
//----------------------------------------------------------------------

// TODO:根据优先级调整规则实现该函数
Thread *
Scheduler::FindNextToRun()
{
    ASSERT(kernel->interrupt->getLevel() == IntOff);

    if (readyList->IsEmpty())
    {
        return NULL;
    }
    else
    {
        //begin: 实验二的代码请写在这里++++++++++++++++++++
        // TODO：
        //更新当前线程的优先级
        //flushPriority();刷新所有就绪线程的优先级
        flushPriority();
        //end: ++++++++++++++++++++++++++++++++++++++++

        Print(); //打印当前进程状态。（助教已实现函数，这个函数不要修改，也不要挪位置,否则可能得不到正确结果)

        //begin: 实验二的代码请写在这里++++++++++++++++++++
        // TODO
        // 计算最近调度与现在的间隔
        // 间隔太小，返回 NULL （避免过分频繁地调度）
        // 就绪队列为空，返回 NULL （没有就绪进程可以调度）
        // 找到优先级最高的就绪态进程t
       Thread *t=readyList->Front();
	Thread *cur=kernel->currentThread;
	int total_ticks=kernel->stats->totalTicks;
        if(total_ticks - lastSwitchTick<MinSwitchPace)
	  return NULL;
	
	if(t->getPriority() <= cur->getPriority())
	    return NULL;
	else 
	{
	lastSwitchTick=kernel->stats->totalTicks;
	ASSERT( ! readyList->IsInList(cur));
	 //ASSERT(kernel->interrupt->getLevel() == IntOff);
	//ASSERT(  readyList->IsInList(cur));
	//if(! readyList->IsInList(cur));
	   // readyList->Insert(cur);
	 //ASSERT( ! readyList->IsInList(cur));
 
        return readyList->RemoveFront();
	}
        //end: ++++++++++++++++++++++++++++++++++++++++
    }
}

//----------------------------------------------------------------------
// Scheduler::Run
// 	Dispatch the CPU to nextThread.  Save the state of the old thread,
//	and load the state of the new thread, by calling the machine
//	dependent context switch routine, SWITCH.
//
//      Note: we assume the state of the previously running thread has
//	already been changed from running to blocked or ready (depending).
// Side effect:
//	The global variable kernel->currentThread becomes nextThread.
//
//	"nextThread" is the thread to be put into the CPU.
//	"finishing" is set if the current thread is to be deleted
//		once we're no longer running on its stack
//		(when the next thread starts running)
//----------------------------------------------------------------------

void Scheduler::Run(Thread *nextThread, bool finishing)
{
    Thread *oldThread = kernel->currentThread;

    ASSERT(kernel->interrupt->getLevel() == IntOff);

    if (finishing)
    { // mark that we need to delete current thread
        ASSERT(toBeDestroyed == NULL);
        toBeDestroyed = oldThread;
    }

    if (kernel->currentThread->space != NULL)
    {                                           // if this thread is a user program,
        kernel->currentThread->SaveUserState(); // save the user's CPU registers
        kernel->currentThread->space->SaveState();
    }

    oldThread->CheckOverflow(); // check if the old thread
                                // had an undetected stack overflow

    kernel->currentThread = nextThread; // switch to the next thread
    nextThread->setStatus(RUNNING);     // nextThread is now running

    DEBUG(dbgThread, "Switching from: " << oldThread->getName() << " to: " << nextThread->getName());

    // This is a machine-dependent assembly language routine defined
    // in switch.s.  You may have to think
    // a bit to figure out what happens after this, both from the point
    // of view of the thread and from the perspective of the "outside world".

    SWITCH(oldThread, nextThread);

    // we're back, running oldThread

    // interrupts are off when we return from switch!
    ASSERT(kernel->interrupt->getLevel() == IntOff);

    DEBUG(dbgThread, "Now in thread: " << oldThread->getName());

    CheckToBeDestroyed(); // check if thread we were running
                          // before this one has finished
                          // and needs to be cleaned up

    if (kernel->currentThread->space != NULL)
    {                                              // if there is an address space
        kernel->currentThread->RestoreUserState(); // to restore, do it.
        kernel->currentThread->space->RestoreState();
    }
}

//----------------------------------------------------------------------
// Scheduler::CheckToBeDestroyed
// 	If the old thread gave up the processor because it was finishing,
// 	we need to delete its carcass.  Note we cannot delete the thread
// 	before now (for example, in Thread::Finish()), because up to this
// 	point, we were still running on the old thread's stack!
//----------------------------------------------------------------------

void Scheduler::CheckToBeDestroyed()
{
    if (toBeDestroyed != NULL)
    {
        delete toBeDestroyed;
        toBeDestroyed = NULL;
    }
}

//----------------------------------------------------------------------
// Scheduler::Print
// 	Print the scheduler state -- in other words, the contents of
//	the ready list.  For debugging.
//----------------------------------------------------------------------
void Scheduler::Print()
{
    fprintf(kernel->thread_sch_f, "\n\n-------------------------one switch-------------------------------\n");
    Thread *t = kernel->currentThread;
    fprintf(kernel->thread_sch_f, "running: tid=%d\t name=%-15s status=%-7s\t pri=%d\n", t->getTid(), t->getName(), statusStr[t->getStatus()], t->getPriority());
    fprintf(kernel->thread_sch_f, "\nReady list contents:\n");
    readyList->Apply(ThreadPrint);
}
