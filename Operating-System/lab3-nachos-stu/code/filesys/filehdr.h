// filehdr.h 
//	Data structures for managing a disk file header.  
//
//	A file header describes where on disk to find the data in a file,
//	along with other information about the file (for instance, its
//	length, owner, etc.)
//
// Copyright (c) 1992-1993 The Regents of the University of California.
// All rights reserved.  See copyright.h for copyright notice and limitation 
// of liability and disclaimer of warranty provisions.

#include "copyright.h"

#ifndef FILEHDR_H
#define FILEHDR_H

#include "disk.h"
#include "pbitmap.h"

/* LAB3 */
const int MaxIndirect = 4;   //max indirect sectors
const int NumInDirectIndex = SectorSize/sizeof(int);   //how many indexes in a index sector
const int NumDirect =	((SectorSize - 3 * sizeof(int) - MaxIndirect*sizeof(int)) / sizeof(int));   //direct
const int DirectSize = NumDirect * SectorSize;    //total direct block size
const int InDirectSectorSize = NumInDirectIndex * SectorSize; //data size that each indirect block can contain
const int MaxFileSize =	DirectSize + InDirectSectorSize * MaxIndirect;
/* ++++ */

// The following class defines the Nachos "file header" (in UNIX terms,  
// the "i-node"), describing where on disk to find all of the data in the file.
// The file header is organized as a simple table of pointers to
// data blocks. 
//
// The file header data structure can be stored in memory or on disk.
// When it is on disk, it is stored in a single sector -- this means
// that we assume the size of this data structure to be the same
// as one disk sector.  Without indirect addressing, this
// limits the maximum file length to just under 4K bytes.
//
// There is no constructor; rather the file header can be initialized
// by allocating blocks for the file (if it is a new file), or by
// reading it from disk.

class FileHeader {
  public:
    bool Allocate(PersistentBitmap *bitMap, int fileSize);// Initialize a file header, 
						//  including allocating space 
						//  on disk for the file data
    void Deallocate(PersistentBitmap *bitMap);  // De-allocate this file's 
						//  data blocks

    void FetchFrom(int sectorNumber); 	// Initialize file header from disk
    void WriteBack(int sectorNumber); 	// Write modifications to file header
					//  back to disk

    int ByteToSector(int offset);	// Convert a byte offset into the file
					// to the disk sector containing
					// the byte

    int FileLength();			// Return the length of the file 
					// in bytes

    void Print();			// Print the contents of the file.

    /* +++LAB3+++ */
    FileHeader();
    int expandFile(int numSec, PersistentBitmap* freeMap);
    void clearIndexTable(int* sectors);
    /* ++++++++++ */

  private:
    int numBytes;			// Number of bytes in the file
    int numSectors;			// Number of data sectors in the file (not include index sectors)
    int dataSectors[NumDirect];		// Disk sector numbers for each data 
					// block in the file
    /* ++++ LAB3 ++++ */
    int numIndirectSectors;   // Number of index sectors in the file
    int indirectSectors[MaxIndirect]; // index sectors
    /* ++++++++++++++ */
};

#endif // FILEHDR_H
