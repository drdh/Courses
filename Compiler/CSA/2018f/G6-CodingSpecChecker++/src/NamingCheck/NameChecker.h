#pragma once

#include "AdaptiveDict.h"

class NameChecker {
public:
    NameChecker(const string &dict_dir, const string &naming_prefix)
    : dict(dict_dir), naming_prefix(naming_prefix) {
    }

    // TODO: rename this to is_valid_macro_name, because it is only intended for macros
    bool is_valid_name(const string &name) {
        regex reg("[a-z]+|[A-Z]+");
        sregex_iterator word_it(name.begin(), name.end(), reg), end_word_it;

        for (; word_it != end_word_it; ++word_it) {
            string lower_word = word_it->str();
            transform(lower_word.begin(), lower_word.end(), lower_word.begin(), ::tolower);

            if (!dict.is_valid(lower_word)) {
                return false;
            }
        }

        return true;
    }

    template <typename T>
    void check(const T *decl) {
        // Don't check decls from system header
        const SourceLocation &location = decl->getLocation();
        const clang::ASTContext &ctx = decl->getASTContext();
        if (ctx.getSourceManager().isInSystemHeader(location)
            || ctx.getSourceManager().isInExternCSystemHeader(location)) {
            return;
        }
        // TODO: temporary solution
        if (!decl->getSourceRange().getBegin().isValid())
            return;

        const string &name = decl->getNameAsString();
        regex reg("[a-z]+|[A-Z]+$|[A-Z][a-z]*");  // TODO: fix the second rule
        sregex_iterator word_it(name.begin(), name.end(), reg), end_word_it;
        
        // Assume the word with prefix and have a try.
        // If it begins with naming prefix, don't check it later.
        // Also, we won't suggest abbr for words with naming-prefix.
        // Note that this is not alway true, but almost.
        if (word_it != end_word_it) {
            string original_word = word_it->str();
            string word_no_prefix;
            auto tmp = std::mismatch(original_word.begin(), original_word.end(), naming_prefix.begin());
            if (!naming_prefix.empty() && tmp.second == naming_prefix.end()) {
                // FIXME: this is slow
                std::copy(tmp.first, original_word.end(), back_inserter(word_no_prefix));
                if (dict.is_valid(word_no_prefix))
                    ++word_it;
            }
        }

        DiagnosticsEngine &DE = decl->getASTContext().getDiagnostics();
        // Note that identifiers are splitted into words,
        // we check all the words and give suggestions for each word if have
        for (; word_it != end_word_it; ++word_it) {
            string lower_word = word_it->str();
            transform(lower_word.begin(), lower_word.end(), lower_word.begin(), ::tolower);

            if (!dict.is_valid(lower_word)) {
                const unsigned ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Warning,
                        "`%0` is neither an English word nor a common abbreviation");
                auto DB = DE.Report(location, ID);
                DB.AddString(word_it->str());
                // record this location, maybe extra analysis later
                auto sit = suspects.find(word_it->str());
                if (sit == suspects.end()) {
                    suspects[word_it->str()] = {location};
                } else {
                    sit->second.push_back(location);
                }
            } else if (true) {
                auto abbr = dict.suggest_abbr(lower_word);
                if (!abbr)
                    continue;
                // Suggest an abbreviation
                const unsigned ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Remark,
                        "`%0` has abbr `%1`");
                auto DB = DE.Report(location, ID);
                DB.AddString(word_it->str());
                DB.AddString(*abbr);
            }
        }
    }

    void check_macro(const string &name) {
        llvm::errs() << "MACRO Name: " << name << '\n';
    }

    AdaptiveDict &get_dict() {
        return dict;
    }
private:
    AdaptiveDict dict;

    // Some config info
    string naming_prefix;
    bool suggest_abbr = true;

    typedef unsigned fuid_t;
    std::unordered_set<fuid_t> headers_this_run;
    std::unordered_set<fuid_t> scanned_headers;

    std::unordered_map<std::string, std::vector<clang::SourceLocation>> suspects;
};