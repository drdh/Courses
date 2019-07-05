#include "AdaptiveDict.h"
#include "llvm/Support/Error.h"

#include <fstream>
#include <regex>


typedef struct {
	int child,brother;
	char c;
	bool end;		
}TrieNode;
