#include "AdaptiveDict.h"

#include "llvm/Support/Error.h"

#include <fstream>
#include <regex>
#include <iostream>
#include <fstream>
#include <string>
#include <string.h>
#include <algorithm>
using namespace std;
//从文件trie.txt中加载Trie树 

TrieNode* AdaptiveDict::load_trie(){
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
//建立trie树，从words.txt中读入词典并把trie树存储在trie.txt中 
TrieNode* AdaptiveDict::build_trie(){
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
		ofstream fout("trie.txt");
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


AdaptiveDict::AdaptiveDict(const std::experimental::filesystem::path &dict_dir) {
    auto load_word_set = [](const std::experimental::filesystem::path &fpath, word_set_t & word_set) {
        ifstream word_set_file(fpath);
        if (!word_set_file) {
            llvm::errs() << "[NamingCheck]: Word dict " << fpath.c_str() << " not found, use empty\n";
        } else {
            string line;
            while (getline(word_set_file, line)) {
                transform(line.begin(), line.end(), line.begin(), ::tolower);
                word_set.insert(line);
            }
        }
    };
    // Load word sets
    load_word_set(dict_dir/"verbs.txt", verb_set);
    load_word_set(dict_dir/"adjs.txt", adj_set);
    load_word_set(dict_dir/"nouns.txt", noun_set);
    load_word_set(dict_dir/"preps.txt", prep_set);
    // Load abbr dict
    build_trie();
    Trie=load_trie();
    //initialize Trie
    {
        ifstream abbr_dict_file(dict_dir/"abbr_dict.txt");
        if (!abbr_dict_file) {
            llvm::errs() << "[NamingCheck]: Abbr dict " << (dict_dir/"abbr_dict.txt").c_str() << " not found, use empty\n";
        } else {
            string line;
            regex word_abbr_delim(": ");
            while (getline(abbr_dict_file, line)) {
                // parse "word: abbr"
                // FIXME: what if wrong format in abbr dict?
                sregex_token_iterator it(line.begin(), line.end(), word_abbr_delim, -1);
                string word = *it;
                string abbr = *(++it);
                abbr_dict[word] = abbr;
                // insert abbr to corresponding word set
                bool classified = false;
                if (match_whole(word, noun_set)) {
                    noun_set.insert(abbr);
                    classified = true;
                }
                if (match_whole(word, verb_set)) {
                    verb_set.insert(abbr);
                    classified = true;
                }
                if (match_whole(word, adj_set)) {
                    adj_set.insert(abbr);
                    classified = true;                    
                }
                if (!classified) {
                    word_bucket.insert(word);
                    word_bucket.insert(abbr);
                }
            }
        }
    }
}

/*
 * Check if given word is valid.
 * The word is valid if and only if is of the form:
 * * Noun
 * * Verb
 * * NounNoun
 * * AdjNoun
 * * VerbNoun
 * And each component can be its corresponding abbr.
 */
bool AdaptiveDict::is_valid(const string &word) {
    // match noun
    if (match_whole(word, noun_set))
        return true;
    // match verb
    if (match_whole(word, verb_set))
        return true;
    // match adj
    if (match_whole(word, adj_set))
        return true;
    // match preps
    if (match_whole(word, prep_set))
        return true;
    // match in word_bucket
    if (match_whole(word, word_bucket))
        return true;
    // // Let's be permissive
    // // noun-noun, noun-verb
    // if (match_concat(word, noun_set, {&noun_set, &verb_set}))
    //     return true;
    // // verb-noun, verb-adj, verb-prep
    // if (match_concat(word, verb_set, {&noun_set, &adj_set, &prep_set}))
    //     return true;
    // // adj-noun
    if (match_concat(word, adj_set, {&noun_set}))
        return true;
    return false;
}

// This is efficient, since a word is not too long.
bool AdaptiveDict::match_concat(const string &word, AdaptiveDict::word_set_t &ws1, vector<AdaptiveDict::word_set_t*> &&ws2s) {
    for (auto it = word.begin()+1; it < word.end(); ++it) {
        // Split original word into two words
        std::string first(word.begin(), it);
        std::string second(it, word.end());
        if (ws1.find(first) != ws1.end()) {
            for (auto ws2: ws2s) {
                if (ws2->find(second) != ws2->end())
                    return true;
            }
        }
    }
    return false;
}

bool AdaptiveDict::match_whole(const string &word, AdaptiveDict::word_set_t &word_set) {
    if (word_set.find(word) != word_set.end())
        return true;
    return false;
}

optional<const string> AdaptiveDict::suggest_abbr(const string &word) {
    auto it = abbr_dict.find(word);
    if (it != abbr_dict.end())
        return optional<const string>(it->second);
    else
        return optional<const string>();
}

//对输入字符串（标识符名），返回合法的表达形式。如果输入中有无法识别的单词，返回nullptr 
string* AdaptiveDict::match_trie(TrieNode* a,string s){	
	int n=s.length();
	for (int i=0;i<n;i++){
		s[i]=tolower(s[i]);
	}//转为小写 
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
