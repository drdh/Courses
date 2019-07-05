////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2018, University of Science and Techonolgy of China
/// All rights reserved.
///
///\file ClassCheck.h
///\brief The tool that check the style of class definition.
///
///\version 0.1.0
///\author ypliu<a href = "linkURL">ypliu88@mail.ustc.edu.cn</a>
////////////////////////////////////////////////////////////////////////////////

#pragma once


#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/MacroInfo.h"
#include "llvm/ADT/StringRef.h"

#include <set>
#include <vector>
#include <iostream>

struct InclusionInfo{
    clang::SourceLocation HashLoc;
    const clang::Token& IncludeTok;
    clang::StringRef FileName;
    clang::StringRef FullPath;
    bool IsAngled;
    clang::SrcMgr::CharacteristicKind FileType;
    InclusionInfo(clang::SourceLocation HashLoc,const clang::Token& IncludeTok, clang::StringRef FileName,
         clang::StringRef FullPath,bool IsAngled, clang::SrcMgr::CharacteristicKind FileType):
         HashLoc(HashLoc),IncludeTok(IncludeTok),FileName(FileName),
         FullPath(FullPath),IsAngled(IsAngled),FileType(FileType){}
};
struct MacroMark{
    llvm::StringRef name;
    bool is_Undef;
    MacroMark(llvm::StringRef name,bool is_Undef):
        name(name),is_Undef(is_Undef){}
};
class PPChecker:public clang::PPCallbacks{

public:
    PPChecker(clang::CompilerInstance &CI,std::vector<std::string>& SourceList)
        :isHeaderFile(false),hasPragmaOnce(false),CI(CI),SourceList(SourceList){
        clang::DiagnosticsEngine& DE = CI.getDiagnostics();
        UNDEF = DE.getCustomDiagID(
            clang::DiagnosticsEngine::Warning, "In file:%0 macro %1 should undef");
        MULDEF = DE.getCustomDiagID(
            clang::DiagnosticsEngine::Warning, "In file %0 macro %1 is used before, you should define another");
        DEF_IN_HEADER = DE.getCustomDiagID(
            clang::DiagnosticsEngine::Warning, "In file %0 macro %1 : macro should not define in header file");
        UNRECOMMENDED = DE.getCustomDiagID(
            clang::DiagnosticsEngine::Warning, "In file %0 macro %1 is unrecommended and maybe %2 is better.");
        DOUBLESHARP = DE.getCustomDiagID(
            clang::DiagnosticsEngine::Warning, "In file %0 macro %1 should not use ## ");
        ORDER = DE.getCustomDiagID(
            clang::DiagnosticsEngine::Warning, "In file %0 the inclusion order is not recommended");
        DOTDIR = DE.getCustomDiagID(
            clang::DiagnosticsEngine::Warning, "%0 inclusion should not use dot dir");
    }

public:
	void InclusionDirective(
		clang::SourceLocation HashLoc,
		const clang::Token &IncludeTok,
		clang::StringRef FileName,
		bool IsAngled,
		clang::CharSourceRange FilenameRange,
		const clang::FileEntry *File,
		clang::StringRef SearchPath,
		clang::StringRef RelativePath,
		const clang::Module *Imported,
		clang::SrcMgr::CharacteristicKind FileType) override;
    
    void EndOfMainFile()override;
    void MacroDefined(const clang::Token &MacroNameTok,
                      const clang::MacroDirective *MD)override;
    void MacroUndefined(const clang::Token &MacroNameTok,
                        const clang::MacroDefinition &MD, 
                        const clang::MacroDirective *Undef)override;
        
public:
    void checkUndef();
    void checkDefineInHeaderFile(const clang::Token &MacroNameTok,
                                const clang::MacroDirective *MD);
    void checkMultiDefine(const clang::Token &MacroNameTok,
                        const clang::MacroDirective *MD);
    void checkHeaderFileProtect();
    void checkUnrecommendedMacro(const clang::Token &MacroNameTok,
                                const clang::MacroDirective *MD);
    void checkDoubleSharpInMacro(const clang::Token &MacroNameTok,
                                const clang::MacroDirective *MD);
    void checkIncludeOrder();
private:
    std::vector<MacroMark>::iterator
        findMacro(llvm::StringRef name);
    void setMacro(llvm::StringRef name){
        auto ptr = findMacro(name);
        if(ptr != macros.end()){
            ptr->is_Undef = true;
        }
    }

    void checkIsNewFile(std::string &loc);
    bool isFileInSource(std::string &file);
    std::string getFullPathFromLocation(const clang::SourceLocation &location);
    void checkIsNewInclusionFile(std::string &file);
    
private:
    clang::CompilerInstance &CI;
    std::vector<std::string>& SourceList;
    std::vector<MacroMark> macros;
    std::vector<InclusionInfo> inclusionInfoSet;
    std::string currFile;
    std::string currInclusionFile;
    bool isHeaderFile;
    bool hasPragmaOnce;
    unsigned int UNDEF;
    unsigned int MULDEF;
    unsigned int DEF_IN_HEADER;
    unsigned int UNRECOMMENDED;
    unsigned int DOUBLESHARP;
    unsigned int ORDER;
    unsigned int DOTDIR;
};