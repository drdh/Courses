#include "AdaptiveDict.h"
#include "Trie.h"
#include "llvm/Support/Error.h"

#include <fstream>
#include <regex>
//#include "ssss.h"
AdaptiveDict::AdaptiveDict(const experimental::filesystem::path &dict_dir) {
    TrieTree=load();//加载trie树
}
string* is_valid(TrieNode* a,string s){	
	int n=s.length();
	for (int i=0;i<n;i++){
		s[i]=tolower(s[i]);
	}
	string* newstr=new string(s);
	bool flag=false; 
	int t=0;
	for (int i=0;i<n;i++){
		if (flag)	(*newstr)[i]=toupper((*newstr)[i]);
		flag=false;
		char c=s[i];
		if (a[t].child==-1){
			if (a[t].end){
				flag=true;
				i--;
				t=0;
				continue;
			}
			else{
				return nullptr;
			} 
		}
		else {
			int p=t; 
			t=a[t].child;
			while (a[t].c!=c){
				if (a[t].brother==-1){
					if (a[p].end){
						flag=true;
						i--;
						t=0;
						break;
					}
					else{
						return nullptr;
					}
				}
				else	t=a[t].brother;
			}					
		}
	}
	return newstr;
}
TrieNode* load(){
	ifstream f("trie.txt");
	int n;
	f>>n;
	TrieNode* a=new TrieNode[n];
	for (int i=0;i<n;i++){
		f>>a[i].child;
		f>>a[i].brother;
		f>>a[i].c;
		f>>a[i].end;
	}
	f.close();
	return a;
}

