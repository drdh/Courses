#pragma once

#include <unordered_set>
#include <unordered_map>
#include <string>
#include <optional>
#include <fstream>
#include <experimental/filesystem>
using namespace std;
struct TrieNode{
	int child,brother;
	char c;
	bool end;		
};
class AdaptiveDict {
    public:
        AdaptiveDict(const experimental::filesystem::path &dict_dir);

        bool is_valid(const string &word);
        optional<const string> suggest_abbr(const string &word);
        string* match_trie(TrieNode *a,string s);
        TrieNode* Trie;
    private:
        typedef unordered_set<string> word_set_t;
        word_set_t noun_set, verb_set, adj_set, prep_set, word_bucket;
        TrieNode* load_trie();
        TrieNode* build_trie();
        typedef unordered_map<string, string> abbr_dict_t;
        abbr_dict_t abbr_dict;

        bool match_whole(const string &word, word_set_t &word_set);
        bool match_concat(const string &word, AdaptiveDict::word_set_t &ws1, vector<AdaptiveDict::word_set_t*> &&ws2s);
};
