#include "PPcheck.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/SmallVector.h"
using namespace clang;
using namespace std;
using namespace SrcMgr;

#if 0

#include <iostream>
#define BEGIN(str) std::cout<<str<<" begin"<<std::endl
#define END(str) std::cout<<str<<" end"<<std::endl
#define COUT(str) std::cout<<str<<std::endl;

#else

#define BEGIN(str)
#define END(str)
#define COUT(str)

#endif
static llvm::StringRef opName[]={"plus","star","minus","slash","plusplus","minusminus",
                        "less","lessless","greater","greatergreater"};
static llvm::StringRef literal[]={"numeric_constant","string_literal"};
static llvm::StringRef idName="identifier";
bool isOp(const char *name){
    int size = sizeof(opName);
    while(size>0){
        size--;
        if(opName[size].equals(name)) return true;
    }
    return false;
}
bool hasSubstr(const char *data, const char *sub){
    int i = 0;
    int p = 0;
    while(data[i] != '\0'){
        if(sub[p]=='\0') return true;
        if(data[i] == sub[p])
            p++;
        else
            p = 0;
        i++;
    }
    if(sub[p] == '\0') return true;
    return false;
}
bool isCompilerDefine(const char *data){
    if(data[0]=='_'&&data[1]=='_') return true;
    if(strcmp(data,"_LP64")==0) return true;
    if(strcmp(data,"unix")==0) return true;
    if(strcmp(data,"linux")==0) return true;
    if(strcmp(data,"_GNU_SOURCE")==0) return true;
    return false;
}
bool isFileProtect(const MacroDirective *MD){
    auto info = MD->getMacroInfo();
    return info->tokens_empty();
}
bool endWith(const char*data, int n1, const char *sub, int n2){
    while(n2 && n1 && data[--n1] == sub[--n2]);
    if(n2 <= 0) return true;
    return false;
}
bool shouldBeConstVar(const ArrayRef<Token> tokens){
    if(tokens.size()==0) return false;
    auto ref_begin = tokens.begin();
    auto ref_end = tokens.end();
    while(ref_begin != ref_end){
        if(idName.equals(ref_begin->getName())) return false;
        if(!(ref_begin->isLiteral()||
            isOp(ref_begin->getName())))
            return false;
        ref_begin++;
    }
    return true;

}
bool shouldBeRef(const ArrayRef<Token> tokens){
    if(tokens.size()==0) return false;
    if(tokens.size()==1 && idName.equals(tokens[0].getName()))
        return true;
    return false;
}
bool shouldBeInline(const ArrayRef<Token> tokens){
    ///foreach macro
    int size = tokens.size();
    if(size>5&&strcmp(tokens[0].getName(),"for")
        &&strcmp(tokens[1].getName(),"l_paren")
        &&strcmp(tokens[size-1].getName(),"r_paren"))
        return false;
    ///use compiler macro
    auto ref_begin = tokens.begin();
    auto ref_end = tokens.end();
    while(ref_begin != ref_end){
        if(idName.equals(ref_begin->getName())){
            auto info = ref_begin->getIdentifierInfo();
            if(isCompilerDefine(info->getName().data()))
                return false;
        }
        ref_begin++;
    }
    return true;
}
bool hasDirDot(StringRef FileName){
    return FileName.contains("./") || FileName.contains("../");
}
int comparePathOrder(StringRef path1, StringRef path2){
    SmallVector<StringRef,20> sep1, sep2;
    path1.split(sep1,"/",-1,false);
    path2.split(sep2,"/",-1,false);
    auto begin1 = sep1.begin();
    auto end1 = sep1.end();
    auto begin2 = sep2.begin();
    auto end2 = sep2.end();
    while(begin1 != end1 && begin2 != end2){
        int cmpResult = begin1->compare(*begin2);
        if(cmpResult != 0) return cmpResult;
        begin1++;
        begin2++;
    }
    if(begin1 == end1 ) return -1;
    else return 1;
}
int compareFileType(CharacteristicKind type1, CharacteristicKind type2){
    return -(type1 - type2);
}
void PPChecker::InclusionDirective(
                    clang::SourceLocation HashLoc,
                    const clang::Token &IncludeTok,
                    clang::StringRef FileName,
                    bool IsAngled,
                    clang::CharSourceRange FilenameRange,
                    const clang::FileEntry *File,
                    clang::StringRef SearchPath,
                    clang::StringRef RelativePath,
                    const clang::Module *Imported,
                    clang::SrcMgr::CharacteristicKind FileType){
    
    auto location = getFullPathFromLocation(HashLoc);
    bool is_source = isFileInSource(location);
    if(!is_source) return;
    checkIsNewFile(location);
    COUT(location);
    COUT(FileName.str());
    inclusionInfoSet.emplace_back(HashLoc,IncludeTok,FileName,File->getName(),IsAngled,FileType);

}
void PPChecker::MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD){
    auto location_str = getFullPathFromLocation(MD->getLocation());
    bool is_source = isFileInSource(location_str);
    if(!is_source) return;
    checkIsNewFile(location_str);
    auto info = MD->getMacroInfo();
    StringRef name = MacroNameTok.getIdentifierInfo()->getName();
    auto data = name.data();
    if(isFileProtect(MD)) return;
    checkDefineInHeaderFile(MacroNameTok, MD);
    if(isHeaderFile) return;
    checkMultiDefine(MacroNameTok,MD);
    checkDoubleSharpInMacro(MacroNameTok, MD);
    checkUnrecommendedMacro(MacroNameTok, MD);
    macros.push_back(MacroMark(name,false));
}
///\note In the guide, the order should be:
///     1.Related header
///     2.C library
///     3.c++ library
///     4.other libraries
///     5.project.h
///But it is hard to decide whether a file is related header.
///And the infomation is little. So the order in this check is:
///     1.user_module
///     2.system_module
///     3.user header
///     4.extern c header
///     5.c header
void PPChecker::checkIncludeOrder(){
    if(inclusionInfoSet.empty()) return;
    if(inclusionInfoSet.size() == 1) return;
    using Kind = CharacteristicKind;
    auto inclusion_begin = inclusionInfoSet.begin();
    auto inclusion_end = inclusionInfoSet.end();
    auto tmp = inclusion_begin;
    DiagnosticsEngine &DE = CI.getDiagnostics();
    const SourceManager &sm = CI.getSourceManager();
    while(tmp != inclusion_end){
        if(hasDirDot(tmp->FileName)){
            DE.Report(DOTDIR)<< tmp->HashLoc.printToString(sm);
            return;
        }
        tmp++;
    }
    auto next = inclusion_begin;
    next++;
    auto compare = [&](Kind type){
        while(inclusion_begin != inclusion_end 
            && inclusion_begin->FileType == type){
            if(next == inclusion_end) return;
            if(next->FileType == inclusion_begin->FileType){
                if(comparePathOrder(inclusion_begin->FileName, next->FileName) > 0){
                    DE.Report(ORDER)<<inclusion_begin->HashLoc.printToString(sm);
                    return;
                }
            }
            inclusion_begin = next;
            next++;
        }
    };
    compare(Kind::C_User_ModuleMap);
    compare(Kind::C_System_ModuleMap);
    compare(Kind::C_User);
    compare(Kind::C_ExternCSystem);
    compare(Kind::C_System);
    if(inclusion_begin != inclusion_end){
        DE.Report(ORDER)<<inclusion_begin->HashLoc.printToString(sm);
    }
}
void PPChecker::MacroUndefined(const Token &MacroNameTok,
                               const MacroDefinition &MD,
                               const MacroDirective *Undef){
    
    //BEGIN("MacroUndefined");
    const MacroDirective* MDR = MD.getLocalDirective();
    if(MDR==nullptr) return;
    auto location_str = getFullPathFromLocation(MDR->getLocation());
    bool is_source = isFileInSource(location_str);
    if(!is_source) return;
    StringRef name = MacroNameTok.getIdentifierInfo()->getName();
    auto data = name.data();
    if(isCompilerDefine(data)) return;
    setMacro(name);
    //END("MacroUndefined");
}
void PPChecker::EndOfMainFile(){
    BEGIN("EndOfMainFile");   
    checkUndef();
    checkIncludeOrder(); 
    END("EndOfMainFile");
}
void PPChecker::checkUndef(){
    auto macro_begin = macros.begin();
    auto macro_end = macros.end();
    DiagnosticsEngine &DE = CI.getDiagnostics();
    while(macro_begin != macro_end){
        if(!macro_begin->is_Undef){
            DE.Report(UNDEF)<<currFile<<macro_begin->name;
        }
        macro_begin++;
    }

}
void PPChecker::checkDefineInHeaderFile(const Token &MacroNameTok,
                                        const MacroDirective *MD){

    if(!isHeaderFile || MD->getMacroInfo()->isUsedForHeaderGuard()) return;
    auto name = MacroNameTok.getIdentifierInfo()->getName();
    DiagnosticsEngine &DE = CI.getDiagnostics();
    DE.Report(DEF_IN_HEADER)<<currFile<<name;
}
void PPChecker::checkMultiDefine(const Token &MacroNameTok,
                                const MacroDirective *MD){
    
    auto name = MacroNameTok.getIdentifierInfo()->getName();
    auto data = name.data();
    auto ptr = findMacro(name);
    auto location = MD->getLocation();
    if(ptr != macros.end()){
        DiagnosticsEngine &DE = CI.getDiagnostics();
        DE.Report(MULDEF)<<location.printToString(CI.getSourceManager())<<ptr->name;
    }
}
void PPChecker::checkHeaderFileProtect(){

}
void PPChecker::checkUnrecommendedMacro(const Token &MacroNameTok,
                                        const MacroDirective *MD){

    //BEGIN("checkUnrecommendedMacro");
    auto info = MD->getMacroInfo();
    if(info->tokens_empty()) return;
    auto params = info->params();
    auto name = MacroNameTok.getIdentifierInfo()->getName();
    auto token_begin = info->tokens_begin();
    auto token_end = info->tokens_end();
    auto tokens = info->tokens();
    auto location = MD->getLocation();
    const SourceManager &sm = CI.getSourceManager();
    DiagnosticsEngine &DE = CI.getDiagnostics();
    if(shouldBeRef(tokens)){
        DE.Report(UNRECOMMENDED)<<location.printToString(sm)<<name<<"reference";
        return;
    }
    if(params.size()==0&&shouldBeConstVar(tokens)){
        DE.Report(UNRECOMMENDED)<<location.printToString(sm)<<name<<"const variable";
        return;
    }
    if(shouldBeInline(tokens)){
        DE.Report(UNRECOMMENDED)<<location.printToString(sm)<<name<<"inline function";
        return;
    }

}
void PPChecker::checkDoubleSharpInMacro(const Token &MacroNameTok,
                                        const MacroDirective *MD){
    
    //BEGIN("checkDoubleSharpInMacro");
    auto info = MD->getMacroInfo();
    auto params = info->params();
    if(info->tokens_empty()) return;
    auto token_begin = info->tokens_begin();
    auto token_end = info->tokens_end();
    DiagnosticsEngine &DE = CI.getDiagnostics();
    auto name = MacroNameTok.getIdentifierInfo()->getName();
    auto location = MD->getLocation();
    const SourceManager &sm = CI.getSourceManager();
    bool preHashHash = false;
    while(token_begin != token_end){
        auto token = token_begin;
        token_begin++;
        if(strcmp(token->getName(),"hashhash")){
            preHashHash = true;
        }
        else if(strcmp(token->getName(),"identifier")&&preHashHash){
            DE.Report(DOUBLESHARP)<<location.printToString(sm)<<name;
            preHashHash = false;
            return;
        }
        else {
            preHashHash = false;
        }
    }
    //END("checkDoubleSharpInMacro");
}

vector<MacroMark>::iterator PPChecker::findMacro(StringRef name){
    auto macro_begin = macros.begin();
    auto macro_end = macros.end();
    while(macro_begin != macro_end){
        if(macro_begin->name.equals(name)){
            return macro_begin;
        }
        macro_begin++;
    }
    return macro_begin;
}
void PPChecker::checkIsNewFile(string& loc){
    
    if(loc.compare(currFile)==0) return;
    checkUndef();
    checkIncludeOrder();
    macros.clear();
    //inclusionInfoSet.clear();
    isHeaderFile = false;
    currFile = loc;
    if(endWith(currFile.data(),currFile.length(),".h",2)||
        endWith(currFile.data(),currFile.length(),".hpp",4))
        isHeaderFile = true;
}
bool PPChecker::isFileInSource(string &file){
    auto source_begin = SourceList.begin();
    auto source_end = SourceList.end();
    while(source_begin != source_end){
        if(source_begin->compare(file)==0)
            return true;
        source_begin++;
    }
    return false;
}
string PPChecker::getFullPathFromLocation(const SourceLocation &location){
    auto loc_str = location.printToString(CI.getSourceManager());
    int cut_index = loc_str.find(':');
    string cut_loc = loc_str.substr(0,cut_index);
    return std::move(cut_loc);
}
void PPChecker::checkIsNewInclusionFile(string &file){
    if(file.compare(currInclusionFile)==0) return;
    checkIncludeOrder();
    currInclusionFile = file;
    inclusionInfoSet.clear();
}

#undef BEGIN
#undef END
#undef COUT