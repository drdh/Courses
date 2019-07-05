////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2018, University of Science and Techonolgy of China
/// All rights reserved.
///
/// @file HeaderCheck.cpp
/// @brief Main entry of this anlysis tool, which parses arguments and dispatches
/// to corresponding FrontendAction instances
///
/// @version 0.1.0
/// @author Shengliang Deng, <a href="linkURL">dengsl@mail.ustc.edu.cn</a> 
/// @author Yuxiang Zhang (Leader), <a href="linkURL">zyx504@mail.ustc.edu.cn</a> 
/// @author Yu Zhang (Mentor), <a href="linkURL">yuzhang@ustc.edu.cn</a> 
////////////////////////////////////////////////////////////////////////////////

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/PPCallbacks.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include "HeaderCheck.h"

#include <set>
#include <iostream>
#include <sstream>

using namespace std;
using namespace clang;

void HeaderCheckASTConsumer::HandleTranslationUnit(ASTContext &Context)
{
}

////////////////////////////////////////////////////////////////////////////////

void DependencyGraphCallback::InclusionDirective(
	SourceLocation HashLoc,
	const Token &IncludeTok,
	StringRef FileName,
	bool IsAngled,
	CharSourceRange FilenameRange,
	const FileEntry *File,
	StringRef SearchPath,
	StringRef RelativePath,
	const Module *Imported,
	SrcMgr::CharacteristicKind FileType)
{

	if (!File)
	{
		return;
	}

	if (FileType != SrcMgr::C_User)
	{
		return;
	}

	SourceManager &SM = CI.getSourceManager();
	const FileEntry *FromFile =
		SM.getFileEntryForID(SM.getFileID(SM.getExpansionLoc(HashLoc)));
	if (!FromFile)
	{
		return;
	}
	Dependencies[FromFile].push_back(File);
	AllFiles.insert(File);
	AllFiles.insert(FromFile);
}

bool DependencyGraphCallback::DFS(const FileEntry *File,
								  ColorSet &Black,
								  ColorSet &Gray,
								  ColorSet &White,
								  CycleList &Cycle)
{
	White.erase(White.find(File));
	Gray.insert(File);
	auto Adjcents = Dependencies[File];
	for (auto I = Adjcents.begin(), E = Adjcents.end(); I != E; I++)
	{
		if (Black.find(*I) != Black.end())
		{
			continue;
		}
		if (Gray.find(*I) != Gray.end())
		{
			Cycle.push_back(*I);
			return true;
		}
		if (DFS(*I, Black, Gray, White, Cycle))
		{
			Cycle.push_back(*I);
			return true;
		}
	}
	Gray.erase(Gray.find(File));
	Black.insert(File);
	return false;
}

void DependencyGraphCallback::EndOfMainFile()
{
	ColorSet Black, Gray, White(AllFiles.begin(), AllFiles.end());
	bool FindCycle = false;
	CycleList Cycle;

	while (!White.empty())
	{
		for (auto I = AllFiles.begin(), E = AllFiles.end(); I != E; I++)
		{
			if (White.find(*I) != White.end())
			{
				if (DFS(*I, Black, Gray, White, Cycle))
				{
					Cycle.push_back(*I);
					ReportCyclicDependencise(Cycle);
					return;
				}
			}
		}
	}
}

void DependencyGraphCallback::ReportCyclicDependencise(CycleList Cycle)
{

	DiagnosticsEngine &DiagEngine = CI.getDiagnostics();
	const unsigned ID = DiagEngine.getCustomDiagID(
		DiagnosticsEngine::Warning, "find cyclic header file inclusion\n%0");

	CI.getDiagnosticClient().BeginSourceFile(CI.getLangOpts(), nullptr);
	std::stringstream SS;
	SS << Cycle[0]->getName().str() << " includes\n";
	for (int I = 1, N = Cycle.size(); I < N; I++)
	{
		SS << Cycle[I]->getName().str() << " includes\n";
	}
	DiagEngine.Report(ID) << SS.str();
	CI.getDiagnosticClient().EndSourceFile();
}
