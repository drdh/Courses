// exception.cc
//  Entry point into the Nachos kernel from user programs.
//  There are two kinds of things that can cause control to
//  transfer back to here from user code:
//
//  syscall -- The user code explicitly requests to call a procedure
//  in the Nachos kernel.  Right now, the only function we support is
//  "Halt".
//
//  exceptions -- The user code does something that the CPU can't handle.
//  For instance, accessing memory that doesn't exist, arithmetic errors,
//  etc.
//
//  Interrupts (which can also cause control to transfer from user
//  code into the Nachos kernel) are handled elsewhere.
//
// For now, this only handles the Halt() system call.
// Everything else core dumps.
//
// Copyright (c) 1992-1996 The Regents of the University of California.
// All rights reserved.  See copyright.h for copyright notice and limitation
// of liability and disclaimer of warranty provisions.

#include "copyright.h"
#include "main.h"
#include "syscall.h"
#include "ksyscall.h"
#define min(a, b) (((a) < (b)) ? (a) : (b))
//----------------------------------------------------------------------
// ExceptionHandler
//  Entry point into the Nachos kernel.  Called when a user program
//  is executing, and either does a syscall, or generates an addressing
//  or arithmetic exception.
//
//  For system calls, the following is the calling convention:
//
//  system call code -- r2
//      arg1 -- r4
//      arg2 -- r5
//      arg3 -- r6
//      arg4 -- r7
//
//  The result of the system call, if any, must be put back into r2.
//
// If you are handling a system call, don't forget to increment the pc
// before returning. (Or else you'll loop making the same system call forever!)
//
//  "which" is the kind of exception.  The list of possible exceptions
//  is in machine.h.
//----------------------------------------------------------------------

void ExceptionHandler(ExceptionType which)
{
    int type = kernel->machine->ReadRegister(2);

    DEBUG(dbgSys, "Received Exception " << which << " type: " << type << "\n");

    switch (which)
    {
    case SyscallException:
        switch (type)
        {
        case SC_Halt:
            DEBUG(dbgSys, "Shutdown, initiated by user program.\n");

            SysHalt();

            ASSERTNOTREACHED();
            break;

        // TODO:处理用户态发来的系统调用请求
        case SC_Exec:
        {

            { //begin: 实验二的代码请写在这里++++++++++++++++++++++++++++++++++++++++

                // 通过调用userprog/ksyscall.h的SysExec函数处理
                // SysExec(fileName);//SysExec()等待你实现
                int buffBase = (int)kernel->machine->ReadRegister(4);
		 int ch;
		int count = 0;
		char *buff = new char[128];
            do
            {
                kernel->machine->ReadMem(buffBase + count, 1, &ch);
                buff[count++] = (char)ch;
            } while ((char)ch != '\0' && count < 128);
	    
	    SysExec(buff);
	    kernel->machine->PCplusPlus();
            } //end: ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            return;
            ASSERTNOTREACHED();
            break;
        }

        // TODO:处理用户态发来的系统调用请求
        case SC_Join:
        {
            { //begin: 实验二的代码请写在这里++++++++++++++++++++++++++++++++++++++++

                // 通过调用当前进程(父进程)的join函数，来等待指定的子进程
                // kernel->currentThread->join(id);//join()等待你实现
               int id = (int)kernel->machine->ReadRegister(4);
	       kernel->currentThread->join(id);
	       kernel->machine->PCplusPlus();
     
            } //end: ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            return;
            ASSERTNOTREACHED();
            break;
        }

        // TODO:处理用户态发来的系统调用请求
        case SC_Fork:
        {

            { //begin: 实验二的代码请写在这里++++++++++++++++++++++++++++++++++++++++

                // 通过调用userprog/ksyscall.h的SysFork函数处理
                int id = SysFork(); //SysFork()等待你实现

                // 后续处理
		kernel->machine->WriteRegister(2, id);
		kernel->machine->PCplusPlus();

            } //end: ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            return;
            ASSERTNOTREACHED();
            break;
        }

        case SC_Read:
        {
            int buffBase = (int)kernel->machine->ReadRegister(4);
            int length = (int)kernel->machine->ReadRegister(5);
            int fileID = (int)kernel->machine->ReadRegister(6);
            //cout<<buffBase<<"  "<<length<<"  "<<fileID<<endl;

            char buff[64];
            int numRead = SysRead(buff, length, fileID);

            {
                //若要运行实验一，需要把你的一部分代码考到这里
            }
            for (int i = 0; i < numRead; i++)
            {
                if (!kernel->machine->WriteMem(buffBase + i, 1, buff[i]))
                    cerr << "ERROR SC_Read: unkown error happened!" << endl;
            }

            kernel->machine->WriteRegister(2, numRead);

            kernel->machine->PCplusPlus();

            return;
            ASSERTNOTREACHED();
            break;
        }

        case SC_Write:
        {
            int buffBase = (int)kernel->machine->ReadRegister(4);
            int length = (int)kernel->machine->ReadRegister(5);
            int fileID = (int)kernel->machine->ReadRegister(6);

            int ch;
            int count = 0;
            char *buff = new char[128];
            do
            {
                kernel->machine->ReadMem(buffBase + count, 1, &ch);
                buff[count++] = (char)ch;
            } while ((char)ch != '\0' && count < 128);

            int numWritten = SysWrite(buff, min(length, strlen(buff)), fileID);
            // cout << "SC_Write: fileID=" << fileID << ", written bytes#=" << numWritten << ", content=" << buff << endl;

            if (numWritten < 0)
            {
                cerr << "ERROR SC_Write: error happened when read file!" << endl;
            }

            kernel->machine->WriteRegister(2, numWritten);

            kernel->machine->PCplusPlus();

            delete[] buff;

            return;
            ASSERTNOTREACHED();
            break;
        }

        case SC_Sub:
        {
            DEBUG(dbgSys, "sub " << kernel->machine->ReadRegister(4) << " - " << kernel->machine->ReadRegister(5) << "\n");

            /* Process SysAdd Systemcall*/
            int result;
            int op1 = (int)kernel->machine->ReadRegister(4);
            int op2 = (int)kernel->machine->ReadRegister(5);
            result = SysSub(op1, op2);
            cout << op1 << " - " << op2 << " = " << result << endl;
            DEBUG(dbgSys, "Sub returning with " << result << "\n");
            /* Prepare Result */
            kernel->machine->WriteRegister(2, (int)result);

            kernel->machine->PCplusPlus();

            return;

            ASSERTNOTREACHED();

            break;
        }

        case SC_Add:
        {
            DEBUG(dbgSys, "Add " << kernel->machine->ReadRegister(4) << " + " << kernel->machine->ReadRegister(5) << "\n");

            /* Process SysAdd Systemcall*/
            int result;
            int op1 = (int)kernel->machine->ReadRegister(4);
            int op2 = (int)kernel->machine->ReadRegister(5);
            result = SysAdd(op1, op2);
            cout << op1 << " + " << op2 << " = " << result << endl;
            DEBUG(dbgSys, "Add returning with " << result << "\n");
            /* Prepare Result */
            kernel->machine->WriteRegister(2, (int)result);

            kernel->machine->PCplusPlus();

            return;

            ASSERTNOTREACHED();

            break;
        }

        case SC_Exit:
        {
            // cout << "exit syscall : currentThread=" << kernel->currentThread->getName() << endl;
            if (strcmp("main", kernel->currentThread->getName()) == 0)
                SysHalt();
            else
                kernel->currentThread->Finish();

            return;
            ASSERTNOTREACHED();
            break;
        }

        default:
            cerr << "Unexpected system call " << type << "\n";
            break;
        }
        break;
    default:
        cerr << "Unexpected user mode exception" << (int)which << "\n";
        break;
    }
    ASSERTNOTREACHED();
}
