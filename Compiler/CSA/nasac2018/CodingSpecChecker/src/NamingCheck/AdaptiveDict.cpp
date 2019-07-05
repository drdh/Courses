#include "AdaptiveDict.h"

#include "llvm/Support/Error.h"

#include <fstream>
#include <regex>

AdaptiveDict::AdaptiveDict(const filesystem::path &dict_dir) {
    auto load_word_set = [](const filesystem::path &fpath, word_set_t & word_set) {
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