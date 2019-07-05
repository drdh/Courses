////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2018, University of Science and Techonolgy of China
/// All rights reserved.
///
/// @file HeaderCheck.h
/// @brief Main entry of this anlysis tool, which parses arguments and dispatches
/// to corresponding FrontendAction instances
///
/// @version 0.1.0
/// @author Shengliang Deng, <a href="linkURL">dengsl@mail.ustc.edu.cn</a> 
/// @author Yuxiang Zhang (Leader), <a href="linkURL">zyx504@mail.ustc.edu.cn</a> 
/// @author Yu Zhang (Mentor), <a href="linkURL">yuzhang@ustc.edu.cn</a> 
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <set>
#include <vector>

class DependencyGraphCallback : public clang::PPCallbacks
{
  private:
	struct DependencyInfo
	{
		const clang::FileEntry *DepentFile;
		clang::SourceLocation DepentLocation;
	};

	using DependencyMap = llvm::DenseMap<
		const clang::FileEntry *,
		llvm::SmallVector<const clang::FileEntry *, 2>>;

	using ColorSet = std::set<const clang::FileEntry *>;
	using CycleList = std::vector<const clang::FileEntry *>;

  public:
	DependencyGraphCallback(clang::CompilerInstance &CI) : CI(CI) {}

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

	void EndOfMainFile() override;

  private:
	void ReportCyclicDependencise(CycleList Cycle);
	bool DFS(const clang::FileEntry *File,
			 ColorSet &Black, ColorSet &Gray, ColorSet &White,
			 CycleList &Cycle);

  private:
	clang::CompilerInstance &CI;
	llvm::SetVector<const clang::FileEntry *> AllFiles;
	DependencyMap Dependencies;
};

class HeaderCheckASTConsumer : public clang::ASTConsumer
{
  public:
	void HandleTranslationUnit(clang::ASTContext &Context) override;
};
