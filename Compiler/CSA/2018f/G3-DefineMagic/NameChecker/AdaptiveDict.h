#pragma once

#include <unordered_set>
#include <unordered_map>
#include <string>
#include <optional>
#include <fstream>
#include "Trie.h"
#include <experimental/filesystem>
using namespace std;

class AdaptiveDict {
    public:
        AdaptiveDict(const experimental::filesystem::path &dict_dir);
		string* match(TrieNode* a,string s);
    private:
        TrieNode* TrieTree;
		
		TrieNode* load();
};
