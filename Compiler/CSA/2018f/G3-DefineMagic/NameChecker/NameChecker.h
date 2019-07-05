#pragma once

#include "AdaptiveDict.h"
#include "string"
#include "Trie.h"
class NameChecker {
public:
    NameChecker(const string &dict_dir, const string &naming_prefix)
    : dict(dict_dir), naming_prefix(naming_prefix) {
    }

    // TODO: rename this to is_valid_macro_name, because it is only intended for macros
    bool is_valid_name(const string &name) {
        
        if (dict.is_valid(name) == nullptr) {
                return false;
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

		//Don't check decls from system header
        const string &name = decl->getNameAsString();
		string* given_name = new string (*name); 
        string* check_advise = is_valid(given_name);

        DiagnosticsEngine &DE = decl->getASTContext().getDiagnostics();
        // Note that identifiers are splitted into words,
        // we check name and give it an advise
		if (check_advise == nullptr){
			const unsigned ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Warning,
                        "`%0` cannot be found in the dictionary");
		}//如果建议是nullptr，说明在trie树中匹配失败
		else if ((*check_advise) != (*name)){
			const unsigned ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Remark,"Advise change the identifier`%0` to `%1`");
		}//如果建议和原始的命名不同，需要发出警告
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
