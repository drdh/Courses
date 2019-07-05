// filehdr.cc
//	Routines for managing the disk file header (in UNIX, this
//	would be called the i-node).
//
//	The file header is used to locate where on disk the
//	file's data is stored.  We implement this as a fixed size
//	table of pointers -- each entry in the table points to the
//	disk sector containing that portion of the file data
//	(in other words, there are no indirect or doubly indirect
//	blocks). The table size is chosen so that the file header
//	will be just big enough to fit in one disk sector,
//
//      Unlike in a real system, we do not keep track of file permissions,
//	ownership, last modification date, etc., in the file header.
//
//	A file header can be initialized in two ways:
//	   for a new file, by modifying the in-memory data structure
//	     to point to the newly allocated data blocks
//	   for a file already on disk, by reading the file header from disk
//
// Copyright (c) 1992-1993 The Regents of the University of California.
// All rights reserved.  See copyright.h for copyright notice and limitation
// of liability and disclaimer of warranty provisions.

#include "copyright.h"

#include "filehdr.h"
#include "debug.h"
#include "synchdisk.h"
#include "main.h"

//----------------------------------------------------------------------
// FileHeader::Allocate
// 	Initialize a fresh file header for a newly created file.
//	Allocate data blocks for the file out of the map of free disk blocks.
//	Return FALSE if there are not enough free blocks to accomodate
//	the new file.
//
//	"freeMap" is the bit map of free disk sectors
//	"fileSize" is the size of bits of free disk sectors
//----------------------------------------------------------------------

/* ++++++++++++++ LAB 3 ++++++++++++++++++++ */
bool FileHeader::Allocate(PersistentBitmap *freeMap, int fileSize)
{
    numBytes = fileSize;
    numSectors = divRoundUp(fileSize, SectorSize);
    numIndirectSectors = ((numSectors - NumDirect) + NumInDirectIndex - 1) / NumInDirectIndex;
    // check if space is enough
    if (freeMap->NumClear() < (numSectors + numIndirectSectors))
        return FALSE;

    int i;
    for (i = 0; i < numSectors && i < NumDirect; i++)
    {
        dataSectors[i] = freeMap->FindAndSet();// 赋值为下一个索引
        // since we checked that there was enough free space,
        // we expect this to succeed
        ASSERT(dataSectors[i] >= 0);
    }

    int doneSec = i;
    for (int j = 0; j < numIndirectSectors && doneSec < numSectors; j++){
        int sectors[NumInDirectIndex];
        // 洞1:begin
        //    direct sectors之后,终于到了indirect sectors!
        //    你需要对indirectSectors进行处理,方法类似于:
        //          dataSectors[i] = freeMap->FindAndSet();
        //    然后,
        //    你需要将sectors用SynchDisk中的某接口写入到indirectSectors[j]中.
        //    这个洞要注意边界条件(如doneSec等)
      indirectSectors[j]=freeMap->FindAndSet();
	int k;
	for(k=0;k<NumInDirectIndex;k++)
	    sectors[k]=-1;
	
	for(k=0;k<NumInDirectIndex && doneSec < numSectors; k++)
	{
	  sectors[k]=freeMap->FindAndSet();
	  doneSec++;
	}
	kernel->synchDisk->WriteSector(indirectSectors[j],(char *)sectors);
        // 洞1:end
    }
    return TRUE;
}
/* ++++++++++++++++++++++++++++++++++ */

//----------------------------------------------------------------------
// FileHeader::Deallocate
// 	De-allocate all the space allocated for data blocks for this file.
//
//	"freeMap" is the bit map of free disk sectors
//----------------------------------------------------------------------

/* +++++++++++++++ LAB3 ++++++++++++ */
void FileHeader::Deallocate(PersistentBitmap *freeMap)
{
    int i;
    for (i = 0; i < numSectors && i < NumDirect; i++)
    {
        ASSERT(freeMap->Test((int)dataSectors[i])); // ought to be marked!
        freeMap->Clear((int)dataSectors[i]);
    }
    int doneSec = i;
    for (int j = 0; j < numIndirectSectors && doneSec < numSectors; j++)
    {
        int sectors[NumInDirectIndex];
	
        // 洞2:begin
        //   为了deallocate,对于indirectSectors[j]我们要读对应的sector,怎么读呢?
        //   然后将将对应的使用的sector进行clear操作.方法在freeMap中
        //   最后,把indirectSectors[j]中sectors的索引clear之后,
        //   还要讲indirectSectors[j]自身clear掉.
	kernel->synchDisk->ReadSector(indirectSectors[j],(char *)sectors);
	int k;
	for(k=0;k<NumInDirectIndex && doneSec < numSectors; k++)
	{
	  ASSERT(freeMap->Test((int)sectors[i]));
	  freeMap->Clear(sectors[k]);
	  doneSec++;
	  //sectors[k]=-1;
	}
	ASSERT(freeMap->Test((int)indirectSectors[i]));
	freeMap->Clear(indirectSectors[j]);
        // 洞2:end
    }
}
/* +++++++++++++++++++++++++++++++ */

//----------------------------------------------------------------------
// FileHeader::FetchFrom
// 	Fetch contents of file header from disk.
//
//	"sector" is the disk sector containing the file header
//----------------------------------------------------------------------

void FileHeader::FetchFrom(int sector)
{
    kernel->synchDisk->ReadSector(sector, (char *)this);
}

//----------------------------------------------------------------------
// FileHeader::WriteBack
// 	Write the modified contents of the file header back to disk.
//
//	"sector" is the disk sector to contain the file header
//----------------------------------------------------------------------

void FileHeader::WriteBack(int sector)
{
    kernel->synchDisk->WriteSector(sector, (char *)this);
}

//----------------------------------------------------------------------
// FileHeader::ByteToSector
// 	Return which disk sector is storing a particular byte within the file.
//      This is essentially a translation from a virtual address (the
//	offset in the file) to a physical address (the sector where the
//	data at the offset is stored).
//
//	"offset" is the location within the file of the byte in question
//----------------------------------------------------------------------

/* +++++ LAB3 +++++ */
int FileHeader::ByteToSector(int offset)
{
    int sectors[NumInDirectIndex]; //direct sectors under inderect sector
    int indirectIndex = (offset - DirectSize) / InDirectSectorSize;
    int index;
    // 洞3:begin
    //    返回offset这个字节所在的sector号码
    //    如果在direct sector的话很简单,就是返回dataSectors的某个索引.
    //    但是怎么判断offset是不是在direct sector呢?
    //    想一想
    //    判断之后,如果不在direct sector的话,怎么取到indirect sector的号码(索引)呢?
    //    这里同样可能用到ReadSector
    //    这个洞需要的控制/判断逻辑很少,主要是计算,
    //    看看你是否民白了多级索引(在这里就是二级索引)
    //    哒哒哒哒哒哒哒哒
    if(offset<DirectSize)
    {
     return dataSectors[(offset/SectorSize)];
    }
    else
    {
      kernel->synchDisk->ReadSector(indirectSectors[indirectIndex],(char *)sectors);
      index=(offset - DirectSize - indirectIndex*InDirectSectorSize)/SectorSize;
      return sectors[index];
    }
    // 洞3:end
}
/* +++++++++++++++++++ */

//----------------------------------------------------------------------
// FileHeader::FileLength
// 	Return the number of bytes in the file.
//----------------------------------------------------------------------

int FileHeader::FileLength()
{
    return numBytes;
}

//----------------------------------------------------------------------
// FileHeader::Print
// 	Print the contents of the file header, and the contents of all
//	the data blocks pointed to by the file header.
//----------------------------------------------------------------------

void FileHeader::Print()
{
    int i, j, k;
    char *data = new char[SectorSize];

    printf("FileHeader contents.  File size: %d.", numBytes);
    printf("\nFile contents:\n");
    //direct block
    /* ++++++++++++++++++ LAB 3 ++++++++++++++ */
    for (i = k = 0; i < NumDirect; i++)
    {
        if (dataSectors[i] == -1 || dataSectors[i] > NumSectors)
            break;
        kernel->synchDisk->ReadSector(dataSectors[i], data);
        for (j = 0; (j < SectorSize) && (k < numBytes); j++, k++)
        {
            if ('\040' <= data[j] && data[j] <= '\176') // isprint    (data[j])
                printf("%c", data[j]);
            else
                printf("\\%x", (unsigned char)data[j]);
        }
    }
    if (numBytes > DirectSize)
    {
        for (i = 0; i < MaxIndirect; i++)
        {
            if (indirectSectors[i] == -1)
                break;
            int sectors[NumInDirectIndex];
            kernel->synchDisk->ReadSector(indirectSectors[i], (char *)sectors);
            for (int l = 0; l < NumInDirectIndex; l++)
            {
                if (sectors[l] == -1)
                    break;
                int secid = sectors[l];
                if (secid == -1 || secid > NumSectors)
                    break;
                kernel->synchDisk->ReadSector(sectors[l], data);
                for (j = 0; (j < SectorSize) && (k < numBytes); j++, k++)
                {
                    if ('\040' <= data[j] && data[j] <= '\176') //    isprint(data[j])
                        printf("%c", data[j]);
                    else
                        printf("\\%x", (unsigned char)data[j]);
                }
            }
        }
    }
    printf("\n");
    /* ++++++++++++++++++++++++++++++++++++ */
    delete[] data;
}

/* +++++ LAB3 +++++ */
void FileHeader::clearIndexTable(int *sectors)
{
    for (int i = 0; i < NumInDirectIndex; i++)
        sectors[i] = -1;
}

FileHeader::FileHeader() : numIndirectSectors(0)
{
    for (int i = 0; i < NumDirect; i++)
        dataSectors[i] = -1;
    for (int i = 0; i < MaxIndirect; i++)
        indirectSectors[i] = -1;
}

// 在write的文件大小超过原本给的大小的时候边长.
//expand file to numsector data sectors
int FileHeader::expandFile(int numSec, PersistentBitmap *freeMap)
{
    // if (numSec <= numSectors)
    //     return 1;
    // 需要indirect sector的数量
    int indirectsNeed;
    if (numSec > NumDirect)
        indirectsNeed = ((numSec - NumDirect) + NumInDirectIndex - 1) / NumInDirectIndex;
    else
        indirectsNeed = 0;
    // 需要的总的sector的数量
    int sectorsNeed = numSec + indirectsNeed - numSectors - numIndirectSectors;
    if (freeMap->NumClear() < sectorsNeed)
        // 没有足够的sector
        return -1;

    int i;
    // allocate direct sectors
    for (i = numSectors; i < NumDirect && i < numSec;)
        if (dataSectors[i] == -1)
        {
            dataSectors[i] = freeMap->FindAndSet();
            i++;
        }

    int doneSec = i;
    // allocate indirect sectors
    for (int j = 0; j < indirectsNeed && doneSec < numSec; j++)
    {
        // 洞4:begin
        //   之前可能用了一些indirect sector,在这个基础上要进行expand
        //   如果indirectSectors[j]==-1的话
        //      那就给这个indirectSectors[j]赋值为下一个索引
        //   需要注意的是,
        //   可能某个indirectSectors[j]!=-1,但是他没有用完对应的block
        //   所以要在这个block里面继续写我们的内容
        //   做法是将indirectSectors[j]读到我们的sectors[]里面,
        //                  (怎么读呢?用什么函数呢?看源码,想一想)
        //   然后将sectors[k] == -1的部分赋值为下一个索引
        //   最后将sectors写入到indirectSectors[j]中.
        //   这个洞要注意边界条件(如doneSec等)
        int sectors[NumInDirectIndex]; //index table
        if (indirectSectors[j] == -1){   
	  indirectSectors[j]=freeMap->FindAndSet();
	  kernel->synchDisk->ReadSector(indirectSectors[j], (char *)sectors);
	  int k;
	  for(k=0;k<NumInDirectIndex;k++)
	    sectors[k]=-1;
	  
	  for(k=0;k<NumInDirectIndex && doneSec < numSec; k++)
	  {
	    sectors[k]=freeMap->FindAndSet();
	    doneSec++;
	  }
	  kernel->synchDisk->WriteSector(indirectSectors[j],(char *)sectors);	  
        }
        else{ // 有东西
          int k;
	  kernel->synchDisk->ReadSector(indirectSectors[j], (char *)sectors);
	  for(k=0;k<NumInDirectIndex && doneSec < numSec; k++)
	  {
	    if(sectors[k]==-1)
	    {
	    sectors[k]=freeMap->FindAndSet();
	    doneSec++;
	    }
	  }
	  kernel->synchDisk->WriteSector(indirectSectors[j],(char *)sectors);
        }
        // 洞4:end
    }

    numBytes = numSec * SectorSize;
    numSectors = numSec;
    numIndirectSectors = indirectsNeed;

    return 1;
}

/* ++++++++++ */