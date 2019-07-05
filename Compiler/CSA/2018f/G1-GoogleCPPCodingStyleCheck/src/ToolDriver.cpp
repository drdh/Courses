////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2018, University of Science and Techonolgy of China
/// All rights reserved.
///
/// @file ToolDriver.cpp
/// @brief Main entry of this anlysis tool, which parses arguments and dispatches
/// to corresponding FrontendAction instances
///
/// @version 0.1.0
/// @author Yuxiang Zhang (Leader), <a href="linkURL">zyx504@mail.ustc.edu.cn</a>
/// @author Shengliang Deng, <a href="linkURL">dengsl@mail.ustc.edu.cn</a>
/// @author Yu Zhang (Mentor), <a href="linkURL">yuzhang@ustc.edu.cn</a>
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
///\note This file is modified for GoogleCPPCodingStyleCheck
////////////////////////////////////////////////////////////////////////////////
#include "clang/AST/ASTConsumer.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "clang/StaticAnalyzer/Core/CheckerRegistry.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"

#include "ClassCheck/ClassCheck.h"
#include "StmtCheck/StmtCheck.h"

#include "PPCheck/PPcheck.h"
#include "DeclCheck/DeclCheck.h"

// #include "StateStore.h"

// #include <filesystem>
#include <experimental/filesystem>
#include <fstream>
#include <set>
#include <vector>
#include <chrono>

using namespace llvm;
using namespace clang;
using namespace clang::driver;
using namespace clang::ento;
using namespace clang::tooling;
using namespace std::chrono;

namespace fs = std::experimental::filesystem;

#if 1

#include <iostream>
#define BEGIN(str) std::cout<<str<<" begin"<<std::endl
#define END(str) std::cout<<str<<" end"<<std::endl
#define COUT(str) std::cout<<str<<std::endl;

#else

#define BEGIN(str)
#define END(str)
#define COUT(str)

#endif


static std::string HelpMessage =
  "\n"
  "-b <build-path> is used to read a compile command database.\n"
  "\n"
  "\tFor example, it can be a CMake build directory in which a file named\n"
  "\tcompile_commands.json exists. Use -DCMAKE_EXPORT_COMPILE_COMMANDS=ON\n"
  "\tCMake option to get this output. When no build path is specified, \n"
  "\tworking directory is used.\n"
  "\n"
  "<project-path> | <source-path> ... specify targets to be checked.\n"
  "\n"
  "\tIf a directory path is provided, the checking target is all files\n"
  "\tthat can under this directory recusively, otherwise it is specifed\n"
  "\tsource path list. All source paths need point into the source tree\n"
  "\tindicated by the compilations database. When nothing is specifed for\n"
  "\tthis option, current working directory is used.\n";

static cl::extrahelp CommonHelp(HelpMessage);
static cl::OptionCategory GoogleCPPCodingStyleChecker("google-cpp-coding-style-checker options");

static cl::opt<std::string>
BuildPath("b", cl::desc("Build directory path"),
  cl::Optional, cl::cat(GoogleCPPCodingStyleChecker), cl::init("."));

static cl::list<std::string>
SourcePath(cl::desc("<Project directory path> | <Source file paths>"),
  cl::Positional, cl::cat(GoogleCPPCodingStyleChecker));
static cl::opt<std::string>
CheckType("c",cl::desc("Select the check type"),
  cl::Optional, cl::cat(GoogleCPPCodingStyleChecker),cl::init("0"));
static cl::opt<bool>
NoClassCheck("no-class-check", cl::desc("don't run ClassChecker"),
  cl::Optional, cl::cat(GoogleCPPCodingStyleChecker),cl::init(false));
static cl::opt<bool>
NoDeclChack("no-decl-check",cl::desc("don't run DeclChecker"),
  cl::Optional, cl::cat(GoogleCPPCodingStyleChecker),cl::init(false));
static cl::opt<bool>
NoStmtCheck("no-stmt-check",cl::desc("don't run StmtChecker"),
  cl::Optional, cl::cat(GoogleCPPCodingStyleChecker), cl::init(false));
static cl::opt<bool>
NoPPcheck("no-pp-check",cl::desc("don't run PPChecker"),
  cl::Optional, cl::cat(GoogleCPPCodingStyleChecker), cl::init(false));

static const char DefaultType = '0';
static const char ClassCheckType = '5';

////////////////////////////////////////////////////////////////////////////////

static std::vector<std::string> SourcePathList;
static std::unique_ptr<CompilationDatabase> Compilations;

///Added by ypliu, to save the source file absolute path
static std::vector<std::string> SourceList;

/// FIXME: Init the allowed source file extension from user defined configuration
static std::set<std::string> AllowedSourceFileExtension = {".cpp", ".h",".cc",".hpp"};

std::string ErrorFunctionListFilePath;
std::string NamingDictPath;

std::vector<double> TimeSpans;

llvm::Error RecursivelyCollectSourceFilePaths(const std::string ProjectPath) {
  std::error_code ErrorCode;
  fs::recursive_directory_iterator Iterator(fs::path(ProjectPath), ErrorCode);
  fs::recursive_directory_iterator EndIterator;
  while (!ErrorCode && Iterator != EndIterator) {
    const fs::path &Path = *Iterator;
    if (fs::is_regular_file(Path, ErrorCode) &&
      AllowedSourceFileExtension.count(Path.extension().string()) != 0) {
      SourcePathList.push_back(Path.string());
    }
    Iterator++;
  }
  if (ErrorCode) {
    return llvm::make_error<llvm::StringError>(
      "[RecursivelyCollectSourceFilePaths]: " + ErrorCode.message(),
      llvm::inconvertibleErrorCode());
  } else {
    return llvm::Error::success();
  }
}
void getSourceFullPath(){
  auto source_begin = SourcePathList.begin();
  auto source_end = SourcePathList.end();
  while(source_begin != source_end){
    fs::path path = fs::path(*source_begin);
    SourceList.push_back(fs::canonical(path));
    source_begin++;
    std::cout<<fs::canonical(path)<<std::endl;
  }
}

llvm::Error ParseCommandLineOptions(int argc, const char **argv) {
  cl::ResetAllOptionOccurrences();
  cl::HideUnrelatedOptions(GoogleCPPCodingStyleChecker);

  std::string ErrorMessage;
  llvm::raw_string_ostream OS(ErrorMessage);

  if (!cl::ParseCommandLineOptions(argc, argv, "", &OS)) {
    OS.flush();
    return llvm::make_error<llvm::StringError>("[ParseCommandLineOptions]: " +
      ErrorMessage, llvm::inconvertibleErrorCode());
  }

  if (BuildPath.empty()) {
    BuildPath = ".";
    llvm::errs() << "[ParseCommandLineOptions]: "
    "No Build path is specified, current working directory is used to search "
    "for compilations database\n";
  }

  Compilations =
  CompilationDatabase::autoDetectFromDirectory(BuildPath, ErrorMessage);
  if (!Compilations) {
    llvm::errs() << "[ParseCommandLineOptions]: "
    "No compilations database is found, running without flags\n";
    Compilations.reset(
      new FixedCompilationDatabase(".", std::vector<std::string>()));
  }

  std::string ProjectPath;

  if (SourcePath.empty()) {
    ProjectPath = ".";
    llvm::errs() << "[ParseCommandLineOptions]: "
    "Neither Project directory path nor source file path list is specified, "
    "current working directory is used as the project directory to collect all "
    "source files\n";
  } else {
    std::vector<std::string> TempSourcePathList = SourcePath;
    std::string FirstSourcePath = TempSourcePathList[0];
    if (fs::is_directory(fs::path(FirstSourcePath))) {
      llvm::errs() << "[ParseCommandLineOptions]: " + FirstSourcePath +
      " is used as project directory to collect all source files\n";
      ProjectPath = FirstSourcePath;
    }
  }
  bool empty = ProjectPath.empty();
  llvm::Error error_code = llvm::Error::success();
  if (empty) {
    llvm::errs() << "[ParseCommandLineOptions]: "
    "Receive source file list from command line\n";
    SourcePathList = SourcePath;
  } else {
    llvm::errs() << "[ParseCommandLineOptions]: "
    "Collect source files from directory " + ProjectPath + "\n";
    error_code = RecursivelyCollectSourceFilePaths(ProjectPath);
  }
  getSourceFullPath();

  return error_code;
}

llvm::Error ToolInitialization(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  // Initialize targets for clang module support.
  // llvm::InitializeAllTargets();
  // llvm::InitializeAllTargetMCs();
  // llvm::InitializeAllAsmPrinters();
  // llvm::InitializeAllAsmParsers();
  return llvm::Error::success();
}

llvm::Error ToolFinalization() {

  return llvm::Error::success();
}

////////////////////////////////////////////////////////////////////////////////

class CodingSpecCheckAction : public ASTFrontendAction {
public:
  bool BeginSourceFileAction(CompilerInstance &CI) override;
  void ExecuteAction() override;
  std::unique_ptr<ASTConsumer> CreateASTConsumer(
    CompilerInstance &Compiler, llvm::StringRef InFile) override;
};

bool CodingSpecCheckAction::BeginSourceFileAction(CompilerInstance &CI) {
  if(!NoPPcheck){
    Preprocessor &PP = CI.getPreprocessor();
    std::unique_ptr<PPChecker> ppChecker(new PPChecker(CI,SourceList));
    PP.addPPCallbacks(std::move(ppChecker));
  }
  return true;
}

void CodingSpecCheckAction::ExecuteAction() {
  high_resolution_clock::time_point Begin = high_resolution_clock::now();
  ASTFrontendAction::ExecuteAction();
  high_resolution_clock::time_point End = high_resolution_clock::now();
}

std::unique_ptr<ASTConsumer> CodingSpecCheckAction::CreateASTConsumer(
  CompilerInstance &Compiler, llvm::StringRef InFile) {

  std::vector<std::unique_ptr<ASTConsumer>> Consumers;

  std::unique_ptr<AnalysisASTConsumer>
  AnalysisConsumer = CreateAnalysisConsumer(Compiler);

  AnalyzerOptionsRef AnalyzerOptions = Compiler.getAnalyzerOpts();
  AnalyzerOptions->InlineMaxStackDepth = 0;

  if(!NoClassCheck){
      AnalysisConsumer->AddCheckerRegistrationFn([] (CheckerRegistry& Registry){
        Registry.addChecker<ClassChecker>("cppchecker.ClassChecker","Google cpp coding style checker");
      });
      AnalyzerOptions->CheckersControlList.push_back({"cppchecker.ClassChecker",true});
  }

  if(!NoDeclChack){
      AnalysisConsumer->AddCheckerRegistrationFn([] (CheckerRegistry& Registry){
        Registry.addChecker<DeclChecker>("cppchecker.DeclChecker","Google cpp coding style checker");
      });
      AnalyzerOptions->CheckersControlList.push_back({"cppchecker.DeclChecker",true});
  }

  if(!NoStmtCheck){
      COUT("CheckType is 6");
      Consumers.push_back(std::move(
        std::unique_ptr<StmtCheckASTConsumer>(new StmtCheckASTConsumer(SourceList))
      ));
  }

  Consumers.push_back(std::move(AnalysisConsumer));

  return llvm::make_unique<MultiplexConsumer>(std::move(Consumers));
}

void OutputTimingInfo() {
  std::ofstream TimingInfo("./timing.csv");
  if (!TimingInfo.is_open()) {
    llvm::report_fatal_error("[CodingSpecChecker]: "
    "Failed to open timing info file");
  }
  for (int I = 0, N = TimeSpans.size(); I != N; I++) {
    TimingInfo << TimeSpans[I] << "\n";
  }
  TimingInfo.close();
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv) {
  high_resolution_clock::time_point Begin = high_resolution_clock::now();

  llvm::Error ParseError = ParseCommandLineOptions(argc, argv);
  if (ParseError) {
    llvm::report_fatal_error("[CodingSpecChecker]:" +
      llvm::toString(std::move(ParseError)));
  }

  llvm::Error InitError = ToolInitialization(argc, argv);
  if (InitError) {
    llvm::report_fatal_error("[CodingSpecChecker]:" +
      llvm::toString(std::move(InitError)));
  }

  ClangTool Tool(*Compilations, SourcePathList);
  // Clear adjusters because -fsyntax-only is inserted by the default chain.
  // Tool.clearArgumentsAdjusters();
  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
  "-w", ArgumentInsertPosition::END));

  std::unique_ptr<FrontendActionFactory> FrontendFactory;
  FrontendFactory = newFrontendActionFactory<CodingSpecCheckAction>();
  Tool.run(FrontendFactory.get());

  high_resolution_clock::time_point End = high_resolution_clock::now();

  llvm::Error FinalizationError = ToolFinalization();
  if (FinalizationError) {
    llvm::report_fatal_error("[CodingSpecChecker]:" +
      llvm::toString(std::move(FinalizationError)));
  }
}

#undef BEGIN
#undef END
#undef COUT
