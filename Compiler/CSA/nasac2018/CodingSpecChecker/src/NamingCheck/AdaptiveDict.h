#pragma once

#include <unordered_set>
#include <unordered_map>
#include <string>
#include <optional>
#include <fstream>
#include <filesystem>
using namespace std;

class AdaptiveDict {
    public:
        AdaptiveDict(const filesystem::path &dict_dir);

        bool is_valid(const string &word);
        optional<const string> suggest_abbr(const string &word);
    private:
        typedef unordered_set<string> word_set_t;
        word_set_t noun_set, verb_set, adj_set, prep_set, word_bucket;

        typedef unordered_map<string, string> abbr_dict_t;
        abbr_dict_t abbr_dict;

        bool match_whole(const string &word, word_set_t &word_set);
        bool match_concat(const string &word, AdaptiveDict::word_set_t &ws1, vector<AdaptiveDict::word_set_t*> &&ws2s);
};