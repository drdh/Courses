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

//----------------------------------------------------------------------
// Scheduler::Scheduler
// 	Initialize the list of ready but not running threads.
//	Initially, no ready threads.
//----------------------------------------------------------------------

Scheduler::Scheduler()
{ 
    readyList = new SortedList<Thread *>(ComparePriority);
    toBeDestroyed = NULL;
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

void
Scheduler::ReadyToRun (Thread *thread)
{
    ASSERT(kernel->interrupt->getLevel() == IntOff);
    DEBUG(dbgThread, "Putting thread on ready list: " << thread->getName());

    thread->setStatus(READY);
    readyList->Insert(thread);
}

/* ++++++++++++++++++++++++++ LAB 2 +++++++++++++++++++++++++++ */
int ComparePriority(Thread* t1, Thread* t2){
    int cmp;
    int t1p = t1->GetPriority();
    int t2p = t2->GetPriority();
    if(t1p>t2p) cmp = -1;
    else if(t1p<t2p) cmp = 1;
    else cmp = 0;
    return cmp;
}

void
Scheduler::SetNewPriority(Thread* t){
    int oldP = t->GetPriority();
    int newP = oldP-(kernel->stats->totalTicks - lastSwitchTick)/100;
    t->SetPriority(newP);
}

void
Scheduler::FlushPriority(){
    ListIterator<Thread *>* iter = new ListIterator<Thread*>(readyList);
    for(;!iter->IsDone();iter->Next()){
        iter->Item()->SetPriority(iter->Item()->GetPriority()+PriorityAdapt);
    }
}
/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */


//----------------------------------------------------------------------
// Scheduler::FindNextToRun
// 	Return the next thread to be scheduled onto the CPU.
//	If there are no ready threads, return NULL.
// Side effect:
//	Thread is removed from the ready list.
//----------------------------------------------------------------------

Thread *
Scheduler::FindNextToRun ()
{
    ASSERT(kernel->interrupt->getLevel() == IntOff);
    /*++++++++++++++++++++++ LAB 2 +++++++++++++++++++++ */
    if (readyList->IsEmpty()) return NULL;
    //dont need to switch
    else {
        SetNewPriority(kernel->currentThread);
        FlushPriority();
        Print();
        //have to switch
        Thread* current = kernel->currentThread;
        if(current->GetStatus()==BLOCKED){
            lastSwitchTick = kernel->stats->totalTicks;
            return readyList->RemoveFront();
        } else if(kernel->stats->totalTicks - lastSwitchTick < MinInterval) {
            return NULL;
        }
        lastSwitchTick = kernel->stats->totalTicks;
        if(readyList->Front()->GetPriority()<current->GetPriority()){
            return NULL;
        } else {
            return readyList->RemoveFront();
        }
    }
    /*++++++++++++++++++++++++++++++++++++++++++++++++++++ */
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

void
Scheduler::Run (Thread *nextThread, bool finishing)
{
    Thread *oldThread = kernel->currentThread;
    
    ASSERT(kernel->interrupt->getLevel() == IntOff);

    if (finishing) {	// mark that we need to delete current thread
         ASSERT(toBeDestroyed == NULL);
	 toBeDestroyed = oldThread;
    }
    
    if (kernel->currentThread->space != NULL) {	// if this thread is a user program,
        kernel->currentThread->SaveUserState(); 	// save the user's CPU registers
	    kernel->currentThread->space->SaveState();
    }
    
    oldThread->CheckOverflow();		    // check if the old thread
					    // had an undetected stack overflow

    kernel->currentThread = nextThread;  // switch to the next thread
    nextThread->setStatus(RUNNING);      // nextThread is now running
    
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

    CheckToBeDestroyed();		// check if thread we were running
					// before this one has finished
					// and needs to be cleaned up
    
    if (kernel->currentThread->space != NULL) {	    // if there is an address space
        kernel->currentThread->RestoreUserState();     // to restore, do it.
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

void
Scheduler::CheckToBeDestroyed()
{
    if (toBeDestroyed != NULL) {
        delete toBeDestroyed;
	toBeDestroyed = NULL;
    }
}
 
//----------------------------------------------------------------------
// Scheduler::Print
// 	Print the scheduler state -- in other words, the contents of
//	the ready list.  For debugging.
//----------------------------------------------------------------------
void
Scheduler::Print()
{
    Thread* t = kernel->currentThread;
    FILE* stats = kernel->getScheduleStatsFile();
    fprintf(stats,"\n\n--------------switch process------------\n");
    fprintf(stats,"running: tid=%d\tname=%-7s\tstatus=%-7s\tpriority=%d\n",t->GetTid(),t->getName(),ThreadStatusStr[t->GetStatus()],t->GetPriority());
    fprintf(stats,"\nReady list contents:\n");
    readyList->Apply(ThreadPrint);
}
