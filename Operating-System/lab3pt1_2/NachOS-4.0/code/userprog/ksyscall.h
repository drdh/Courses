/**************************************************************
 *
 * userprog/ksyscall.h
 *
 * Kernel interface for systemcalls 
 *
 * by Marcus Voelp  (c) Universitaet Karlsruhe
 *
 **************************************************************/

#ifndef __USERPROG_KSYSCALL_H__ 
#define __USERPROG_KSYSCALL_H__ 

#include "kernel.h"




void SysHalt()
{
  kernel->interrupt->Halt();
}


int SysAdd(int op1, int op2)
{
  return op1 + op2;
}

/* +++++++++++++++++++++++++++LAB1 BEGIN+++++++++++++++++++++++ */

bool SysCreate(char* filename){
  #ifdef FILESYS_STUB
  return kernel->fileSystem->Create(filename);
  #else
  return kernel->fileSystem->Create(filename,0);
  #endif
}

int SysOpen(char* filename){
  OpenFile* file = kernel->fileSystem->Open(filename);
  if(file==NULL) {
    printf("open %s error\n",filename);
    return -1;
  }
  return file->getId();
}

int SysClose(int fileId){
  #ifndef FILESYS_STUB
  return kernel->fileSystem->Close(fileId);
  #else
  return kernel->fileSystem->removeFile(fileId);
  #endif
}

int SysWrite(char *buffer, int size, OpenFileId fid){
  //standard io
  if(fid==1 || fid==2){
    WriteFile(fid,buffer,size);
    return size;
  }
  return kernel->fileSystem->Write(buffer, size, fid);
}

int SysRead(char *buffer, int size, OpenFileId fid){
  //standard io
  if(fid==0)
    return ReadPartial(fid, buffer, size);
  return kernel->fileSystem->Read(buffer, size, fid);
}

int SysSeek(int pos, OpenFileId id){
  OpenFile* file = kernel->fileSystem->getFile(id);
  if(file==NULL){
    printf("file did not open\n");
    return -1;
  }
  file->Seek(pos);
  return 1;                         //return 1 if true
}

/* ++++++++++++++++++++++++++++++LAB1 END++++++++++++++++++++++++++++ */

/* ++++++++++++++++++++++++++++++LAB2 BEGIN++++++++++++++++++++++++++ */

//for fork(), only child process implement this
void childProcedure(){
  kernel->currentThread->RestoreUserState();         //restore register state
  kernel->currentThread->space->RestoreState();      //restore userprog memory to machine memory

  kernel->machine->WriteRegister(2,0);
  kernel->machine->increasePC();

  kernel->machine->Run();
}

void SysExec(char* filename){
  kernel->currentThread->SetName(filename);
  kernel->currentThread->space->Reset();
  if(kernel->currentThread->space->Load(filename)){
    kernel->currentThread->space->Execute();
    ASSERTNOTREACHED();
  }
  else {
    printf("can't exec program %s\n",filename);
    kernel->currentThread->Finish();
  }
}

int SysFork(){
  Thread* child = new Thread("forked");
  AddrSpace* space = new AddrSpace();
  space->CopyMemory(kernel->currentThread->space);
  child->space = space;
  child->SaveUserState();                           //save register state

  child->Fork((VoidFunctionPtr)childProcedure, NULL);
  return child->GetTid();
}

void SysJoin(int tid){
  kernel->currentThread->Join(tid);
}

/* ++++++++++++++++++++++++++++++LAB2 END++++++++++++++++++++++++++++ */

/* ++++ LAB3 ++++ */
void SysPrint(){
  #ifndef SYSFILE_STUB
  kernel->fileSystem->Print();
  #else
  printf("using unix file system\n");
  #endif
}
/* ++++++++++++++ */

#endif /* ! __USERPROG_KSYSCALL_H__ */
