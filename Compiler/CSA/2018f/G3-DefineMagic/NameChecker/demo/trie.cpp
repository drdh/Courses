#include <iostream>
#include <fstream>
#include <string>
#include <string.h>
#include <algorithm>
using namespace std;

struct TrieNode{
	int child,brother;
	char c;
	bool end;		
};
TrieNode* btree;
TrieNode* ftree;
//字符串逆序 
void reverse(string* s){
	int n=s->length();
	for (int i=0;i<n/2;i++){
		char temp=(*s)[i];
		(*s)[i]=(*s)[n-1-i];
		(*s)[n-1-i]=temp;		
	}
}
//建立正向的trie树，从words.txt中读入词典并把trie树存储在trie_f.txt中 
TrieNode* build_f(){
	TrieNode* a=new TrieNode[1000000];
	string* s=new string[140000];
	ifstream f("words.txt");
	int len=1;
	if (f.is_open()){
		int n;
		for (n=0;!f.eof();n++){
			f>>s[n];
		}//读入单词 
		sort(s,s+n);
		a[0]={-1,-1,'$',0};//起始结点 
		for (int i=0;i<n;i++){
			int t=0;
			int m=s[i].length();
			for (int j=0;j<m;j++){
				char c=s[i][j];
				if (a[t].child==-1){
					a[t].child=len;
					//cout<<len<<"\n";
					a[len]={-1,-1,c,0};
					t=len;
					len++;
				}//如果trie树下一层没有结点，就创建一个 
				else {
					t=a[t].child;
					while (a[t].c!=c){
						if (a[t].brother==-1){
							a[t].brother=len;
							a[len]={-1,-1,c,0};
							t=len;
							len++;
						}//找遍了trie树的这一层，没有找到对应的结点，就创建一个 
						else	t=a[t].brother;//找当前结点的弟弟 
					}					
				}
			}
			a[t].end=true;//单词结尾增加终结标记			 
		}//对每一个单词建立trie树 
		f.close();
		ofstream fout("trie_f.txt");
		fout<<len<<"\r\n";
		for (int i=0;i<len;i++){
			fout<<a[i].child<<" ";
			fout<<a[i].brother<<" ";
			fout<<a[i].c<<" ";
			fout<<a[i].end<<"\r\n";
		}
		fout.close();
		//将trie树存储在文件中 
	}
	else {
		cout<<"Input File Is Not Existed\n";
	}	
	return a;
}
//建立反向的trie树，从words.txt中读入词典并把trie树存储在trie_b.txt中 
TrieNode* build_b(){
	TrieNode* a=new TrieNode[1000000];
	string* s=new string[140000];
	ifstream f("words.txt");
	int len=1;
	if (f.is_open()){
		int n;
		for (n=0;!f.eof();n++){
			f>>s[n];
			reverse(&(s[n]));
		}//读入单词 
		sort(s,s+n);
		a[0]={-1,-1,'$',0};//起始结点 
		for (int i=0;i<n;i++){
			int t=0;
			int m=s[i].length();
			for (int j=0;j<m;j++){
				char c=s[i][j];
				if (a[t].child==-1){
					a[t].child=len;
					//cout<<len<<"\n";
					a[len]={-1,-1,c,0};
					t=len;
					len++;
				}//如果trie树下一层没有结点，就创建一个 
				else {
					t=a[t].child;
					while (a[t].c!=c){
						if (a[t].brother==-1){
							a[t].brother=len;
							a[len]={-1,-1,c,0};
							t=len;
							len++;
						}//找遍了trie树的这一层，没有找到对应的结点，就创建一个 
						else	t=a[t].brother;//找当前结点的弟弟 
					}					
				}
			}
			a[t].end=true;//单词结尾增加终结标记			 
		}//对每一个单词建立trie树 
		f.close();
		ofstream fout("trie_b.txt");
		fout<<len<<"\r\n";
		for (int i=0;i<len;i++){
			fout<<a[i].child<<" ";
			fout<<a[i].brother<<" ";
			fout<<a[i].c<<" ";
			fout<<a[i].end<<"\r\n";
		}
		fout.close();
		//将trie树存储在文件中 
	}
	else {
		cout<<"Input File Is Not Existed\n";
	}	
	return a;
}
//对输入的正向字符串（标识符名），采用正向最大匹配，返回合法的表达形式。如果输入中有无法识别的单词，返回nullptr 
string* fmm(TrieNode* a,string &s){	
	int n=s.length();
	string* newstr=new string(s);	
	bool flag=false;//标志当前字符是否要大写 
	int t=0;//虚拟指针，指示当前访问的TrieNode 
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
			}//如果在trie树中找不到当前字符，且前一个字符已经可以作为单词结尾。那么，就将当前字符大写，在trie树中重新开始搜索 
			else{
				return nullptr;
			}//如果前一个字符不可以作为单词结尾，那么，认为存在未知的单词 
		}//如果trie树下一层没有结点
		else {
			int p=t;//记录trie树中的前一个结点 
			t=a[t].child;
			while (a[t].c!=c){
				if (a[t].brother==-1){
					if (a[p].end){
						flag=true;
						i--;
						t=0;
						break;
					}//如果前一个字符已经可以作为单词结尾。那么，就将当前字符大写，在trie树中重新开始搜索 
					else{
						return nullptr;
					}
				}//找遍了trie树的这一层，没有找到对应的结点
				else	t=a[t].brother;//找当前结点的弟弟 
			}					
		}
	}
	return newstr;
}
//对输入的反向字符串（标识符名），采用反向最大匹配，返回合法的表达形式。如果输入中有无法识别的单词，返回nullptr 
string* bmm(TrieNode* a,string &s){	
	int n=s.length();
	string* newstr=new string(s);	
	int t=0;//虚拟指针，指示当前访问的TrieNode 
	for (int i=0;i<n;i++){
		char c=s[i];
		if (a[t].child==-1){
			if (a[t].end){
				(*newstr)[i-1]=toupper((*newstr)[i-1]);
				i--;
				t=0;
				continue;
			}//如果在trie树中找不到当前字符，且前一个字符已经可以作为单词结尾。那么，就将前一个字符大写，在trie树中重新开始搜索 
			else{
				return nullptr;
			}//如果前一个字符不可以作为单词结尾，那么，认为存在未知的单词 
		}//如果trie树下一层没有结点
		else {
			int p=t;//记录trie树中的前一个结点 
			t=a[t].child;
			while (a[t].c!=c){
				if (a[t].brother==-1){
					if (a[p].end){
						(*newstr)[i-1]=toupper((*newstr)[i-1]);
						i--;
						t=0;
						break;
					}//如果前一个字符已经可以作为单词结尾。那么，就将前一个字符大写，在trie树中重新开始搜索 
					else{
						return nullptr;
					}
				}//找遍了trie树的这一层，没有找到对应的结点
				else	t=a[t].brother;//找当前结点的弟弟 
			}					
		}
	}
	reverse(newstr);//将字符串翻转，恢复为正向 
	(*newstr)[0]=tolower((*newstr)[0]);//首字母小写 
	return newstr;
}
//从文件中加载Trie树 
TrieNode* load(string s){
	ifstream f(s);
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
//对输入的字符串（标识符名），采用双向最大匹配的方法（先使用BMM，再使用FMM），返回合法的表达形式。如果输入中有无法识别的单词，返回nullptr  
string* bm(TrieNode* ft,TrieNode* bt,string &t){
	string s=t;
	int n=s.length();
	for (int i=0;i<n;i++){
		s[i]=tolower(s[i]);
	}//转为小写 
	reverse(&s);
	string* ans=bmm(bt,s);
	if (ans!=nullptr)	return ans;
	reverse(&s);
	return fmm(ft,s);
}
//从输入中读入字符串（标识符名），输出其合法形式 
int main() {
	//build_b();
	//build_f();
	ftree=load("trie_f.txt");
	btree=load("trie_b.txt");
	while (true){
		string s;
		cin>>s;
		string* temp=bm(ftree,btree,s);
		if (temp!=nullptr){
			cout<<(*temp)<<"\n";
		}
		else cout<<"Unknown Expression\n";
	}
} 

