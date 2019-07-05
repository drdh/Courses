// exception.cc 
//	Entry point into the Nachos kernel from user programs.
//	There are two kinds of things that can cause control to
//	transfer back to here from user code:
//
//	syscall -- The user code explicitly requests to call a procedure
//	in the Nachos kernel.  Right now, the only function we support is
//	"Halt".
//
//	exceptions -- The user code does something that the CPU can't handle.
//	For instance, accessing memory that doesn't exist, arithmetic errors,
//	etc.  
//
//	Interrupts (which can also cause control to transfer from user
//	code into the Nachos kernel) are handled elsewhere.
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
#include "unistd.h"
//----------------------------------------------------------------------
// ExceptionHandler
// 	Entry point into the Nachos kernel.  Called when a user program
//	is executing, and either does a syscall, or generates an addressing
//	or arithmetic exception.
//
// 	For system calls, the following is the calling convention:
//
// 	system call code -- r2
//		arg1 -- r4
//		arg2 -- r5
//		arg3 -- r6
//		arg4 -- r7
//
//	The result of the system call, if any, must be put back into r2. 
//
// If you are handling a system call, don't forget to increment the pc
// before returning. (Or else you'll loop making the same system call forever!)
//
//	"which" is the kind of exception.  The list of possible exceptions 
//	is in machine.h.
//----------------------------------------------------------------------

/*read string para from [register] to line*/
void readStrPara(char *line, int reg,int size)
{
	int strAdr = kernel->machine->ReadRegister(reg);
	int c;
	for(int idx=0; idx<size; idx++)
	{
		kernel->machine->ReadMem(strAdr+idx,1,&c);
		line[idx]=(char)c;
		if((char)c=='\0') break;
	}
}

void
ExceptionHandler(ExceptionType which)
{
    int type = kernel->machine->ReadRegister(2);

    DEBUG(dbgSys, "Received Exception " << which << " type: " << type << "\n");

    switch (which) 
	{
    	case SyscallException:
        switch(type) 
		{
        	case SC_Halt:
			{
			DEBUG(dbgSys, "Shutdown, initiated by user program.\n");

			SysHalt();

			ASSERTNOTREACHED();
			break;
			}

      		case SC_Add:
			{
			DEBUG(dbgSys, "Add " << kernel->machine->ReadRegister(4) << " + " << kernel->machine->ReadRegister(5) << "\n");
	
			/* Process SysAdd Systemcall*/
			int result;
			result = SysAdd(/* int op1 */(int)kernel->machine->ReadRegister(4),
			/* int op2 */(int)kernel->machine->ReadRegister(5));

			DEBUG(dbgSys, "Add returning with " << result << "\n");
			/* Prepare Result */
			kernel->machine->WriteRegister(2, (int)result);
	
			/* Modify return point */
			kernel->machine->increasePC();

			return;
	
			ASSERTNOTREACHED();

			break;
			}
			
			/*+++++++++++++++++++++++++++++++++LAB1 BEGIN+++++++++++++++++++++++++++++++*/

			/* system call Create(char*) */
	  		case SC_Create:
			{
			char* cFileName;
			cFileName = new char[128];
			readStrPara(cFileName, 4, 128);                          //get cFileName para
			if(SysCreate(cFileName))
				kernel->machine->WriteRegister(2, 1);            //if success, return 1
			else 
			{
				printf("create file %s failed\n",cFileName);
				kernel->machine->WriteRegister(2, -1);
			}
			kernel->machine->increasePC();
			delete[] cFileName;
			return;
			ASSERTNOTREACHED();
			break;
			}

			case SC_Open:
			{
			char* oFileName;
			oFileName = new char[128];
			readStrPara(oFileName, 4, 128);
			int openfile;                  //file id
			openfile = SysOpen(oFileName);
			if(openfile==-1) printf("open %s failed!\n",oFileName);
			kernel->machine->WriteRegister(2,openfile);
			kernel->machine->increasePC();
			delete[] oFileName;
			return;
			ASSERTNOTREACHED();
			break;
			}

			case SC_Close:
			{
			int fileId;
			fileId = kernel->machine->ReadRegister(4);
			if(SysClose(fileId)==-1) printf("close file failed!\n");
			kernel->machine->WriteRegister(2, 1);
			kernel->machine->increasePC();
			return;
			ASSERTNOTREACHED();
			break;
			}

			case SC_Write:
			{
			int size;
			int fileId;
			char* str;
			size = kernel->machine->ReadRegister(5);
			fileId = kernel->machine->ReadRegister(6);
			str = new char[size];
			readStrPara(str, 4, size);
			int writtenSize = SysWrite(str, size, fileId);
			if(writtenSize<0) printf("write failed!\n");
			kernel->machine->WriteRegister(2, writtenSize);
			kernel->machine->increasePC();
			delete[] str;
			return;
			ASSERTNOTREACHED();
			break;
			}

			case SC_Read:
			{
			int strAdr = kernel->machine->ReadRegister(4);
			int size = kernel->machine->ReadRegister(5);
			int fileId = kernel->machine->ReadRegister(6);
			char* tmp = new char[size];
			int read = SysRead(tmp,size,fileId);
			if(read<0) printf("read failed\n");
			else
			{
			for(int i=0;i<read;i++)
				kernel->machine->WriteMem(strAdr+i,1,tmp[i]);
			}
			kernel->machine->WriteRegister(2, read);
			kernel->machine->increasePC();
			return;
			ASSERTNOTREACHED();
			break;
			}

			case SC_Seek:
			{
			int pos = kernel->machine->ReadRegister(4);
			int fileId = kernel->machine->ReadRegister(5);
			int seekResult = SysSeek(pos,fileId);
			if(seekResult==-1) printf("seek error\n");
			kernel->machine->WriteRegister(2, seekResult);
			kernel->machine->increasePC();
			return;
			ASSERTNOTREACHED();
			break;
			}

			/* ++++++++++++++++++++++++++++++++++++LAB1 END++++++++++++++++++++++++++++*/

			/* ++++++++++++++++++++++++++++++++++++LAB2 BEGIN++++++++++++++++++++++++++*/

			case SC_Fork:
			{
			int forkResult = SysFork();
			kernel->machine->WriteRegister(2, forkResult);
			kernel->machine->increasePC();
			return;
			ASSERTNOTREACHED();
			break;
			}

			case SC_Exec:
			{
			char* filename = new char[128];
			readStrPara(filename, 4, 128);
			SysExec(filename);
			kernel->machine->increasePC();

			return;
			ASSERTNOTREACHED();
			break;
			}

			case SC_Join:
			{
			int tid = kernel->machine->ReadRegister(4);
			SysJoin(tid);
			kernel->machine->increasePC();
			return;
			ASSERTNOTREACHED();
			break;
			}

			case SC_Exit:
			{
				//exit main thread
				if(strcmp("main",kernel->currentThread->getName()) == 0){
					SysHalt();
				}

				kernel->currentThread->Finish();

				return;
				ASSERTNOTREACHED();
				break;
			}

			/* ++++++++++++++++++++++++++++++++++++LAB2 END++++++++++++++++++++++++++++*/

			case SC_Print:
			{
				SysPrint();
				kernel->machine->increasePC();
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

