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

#include "ErrorHandlingCheck/ErrorHandlingCheck.h"
#include "FullCommentCheck/FullCommentCheck.h"
#include "HeaderCheck/HeaderCheck.h"
#include "InitInNeedCheck/InitInNeedCheck.h"
#include "ModuleCheck/ModuleCheck.h"
#include "NamingCheck/NamingCheck.h"

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

static std::string HelpMessgae =
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

static cl::extrahelp CommonHelp(HelpMessgae);
static cl::OptionCategory CodingSpecChecker("code-spec-checker options");

static cl::opt<std::string> 
BuildPath("b", cl::desc("Build directory path"), 
  cl::Optional, cl::cat(CodingSpecChecker), cl::init("."));

static cl::list<std::string> 
SourcePath(cl::desc("<Project directory path> | <Source file paths>"), 
  cl::Positional, cl::cat(CodingSpecChecker));

static cl::opt<bool>
NoErrorHandlingCheck("no-error-handling-check", cl::cat(CodingSpecChecker),
  cl::desc("Do not check whether error returned by function call is handled."));

static cl::opt<bool>
NoFullCommentCheck("no-full-comment-check", cl::cat(CodingSpecChecker),
  cl::desc("Do not check whether function header comment is valid."));

static cl::opt<bool>
NoHeaderCheck("no-header-check", cl::cat(CodingSpecChecker), cl::desc("Do not" 
  "check whether header file is self-contained and has no cyclic inclusion."));

static cl::opt<bool>
NoInitInNeedCheck("no-init-in-need-check", cl::cat(CodingSpecChecker),
  cl::desc("Do not check whether variable is initialized in need"));

static cl::opt<bool>
NoModuleCheck("no-module-check", cl::cat(CodingSpecChecker), cl::desc("Do not"
  "check whether a module follows principles of contractual programming"));

static cl::opt<bool>
NoNamingCheck("no-naming-check", cl::cat(CodingSpecChecker),
  cl::desc("Do not check whether identifiers use unaccepted abbrivations"));

static cl::opt<std::string>
NamingDictPathOption("naming-dict-directory", cl::cat(CodingSpecChecker),
	cl::desc("The directory containing word_dict.txt and abbr_dict.txt"),
	cl::Optional);

static cl::opt<std::string>
NamingPrefixOption("naming-prefix", cl::cat(CodingSpecChecker),
  cl::desc("The naming prefix used in this project"),
  cl::Optional);

static cl::opt<std::string>
ErrorFunctionListFilePathOption("error-function-list", cl::cat(CodingSpecChecker),
	cl::desc("The path to list file of functions need error handling"),
  cl::Optional);

static cl::opt<bool>
TurnOnTiming("timing", cl::cat(CodingSpecChecker),
  cl::desc("Timing two elapsed times, i.e Total, ASTAction"), cl::Optional);

////////////////////////////////////////////////////////////////////////////////

static std::vector<std::string> SourcePathList;
static std::unique_ptr<CompilationDatabase> Compilations;

/// FIXME: Init the allowed source file extension from user defined configuration
static std::set<std::string> AllowedSourceFileExtension = {".c", ".h"};

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

llvm::Error ParseCommandLineOptions(int argc, const char **argv) {
  cl::ResetAllOptionOccurrences();
  cl::HideUnrelatedOptions(CodingSpecChecker);

  std::string ErrorMessage;
  llvm::raw_string_ostream OS(ErrorMessage);

  if (!cl::ParseCommandLineOptions(argc, argv, "", &OS)) {
    OS.flush();
    return llvm::make_error<llvm::StringError>("[ParseCommandLineOptions]: " +
      ErrorMessage, llvm::inconvertibleErrorCode());
  }

  if (NamingDictPathOption.empty()) {
    NamingDictPath = filesystem::absolute(
      filesystem::path(argv[0]).parent_path()/".."/"share"/"ccheck"/"dicts").string();
  } else {
    NamingDictPath = NamingDictPathOption;
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

  if (ProjectPath.empty()) {
    llvm::errs() << "[ParseCommandLineOptions]: "
    "Receive source file list from command line\n";
    SourcePathList = SourcePath;
  } else {
    llvm::errs() << "[ParseCommandLineOptions]: "
    "Collect source files from directory " + ProjectPath + "\n";
    return RecursivelyCollectSourceFilePaths(ProjectPath);
  }

  if (ErrorFunctionListFilePathOption.empty()) {
    ErrorFunctionListFilePath = filesystem::absolute(
      filesystem::path(argv[0]).parent_path()/".."/"share"/"ccheck"/"dicts").string();
  } else {
    ErrorFunctionListFilePath = ErrorFunctionListFilePathOption;
  }

  return llvm::Error::success();
}

llvm::Error ToolInitialization(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  // Initialize targets for clang module support.
  // llvm::InitializeAllTargets();
  // llvm::InitializeAllTargetMCs();
  // llvm::InitializeAllAsmPrinters();
  // llvm::InitializeAllAsmParsers();

  NamingCheckASTConsumer::StaticInit(NamingDictPath, NamingPrefixOption);
  ErrorHandlingChecker::StaticInit(ErrorFunctionListFilePath);

  return llvm::Error::success();
}

llvm::Error ToolFinalization() {
  NamingCheckASTConsumer::StaticFinalization();

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
  if (!NoHeaderCheck) {
    Preprocessor &PP = CI.getPreprocessor();
    std::unique_ptr<DependencyGraphCallback>
    DependencyGraphCheck(new DependencyGraphCallback(CI));
    PP.addPPCallbacks(std::move(DependencyGraphCheck));
  }
  return true;
}

void CodingSpecCheckAction::ExecuteAction() {
  high_resolution_clock::time_point Begin = high_resolution_clock::now();
  ASTFrontendAction::ExecuteAction();
  high_resolution_clock::time_point End = high_resolution_clock::now();
  if (TurnOnTiming) {
    double TimeSpan = duration_cast<duration<double>>(End - Begin).count();
    TimeSpans.push_back(TimeSpan);
  }
}

std::unique_ptr<ASTConsumer> CodingSpecCheckAction::CreateASTConsumer(
  CompilerInstance &Compiler, llvm::StringRef InFile) {

  std::vector<std::unique_ptr<ASTConsumer>> Consumers;
  if (!NoHeaderCheck) {
    Consumers.push_back(std::move(
      std::unique_ptr<HeaderCheckASTConsumer>(new HeaderCheckASTConsumer)));
  }
  if (!NoNamingCheck) {
    Consumers.push_back(std::move(
  	  std::unique_ptr<NamingCheckASTConsumer>(new NamingCheckASTConsumer(Compiler.getPreprocessorPtr()))));
  }
  if (!NoFullCommentCheck) {
    Consumers.push_back(std::move(
  	  std::unique_ptr<FullCommentCheckASTConsumer>(new FullCommentCheckASTConsumer)));
  }

  std::unique_ptr<AnalysisASTConsumer> 
  AnalysisConsumer = CreateAnalysisConsumer(Compiler);
  
  AnalyzerOptionsRef AnalyzerOptions = Compiler.getAnalyzerOpts();
  AnalyzerOptions->InlineMaxStackDepth = 0;

  if (!NoModuleCheck) {
    AnalysisConsumer->AddCheckerRegistrationFn([] (CheckerRegistry& Registry) {
      Registry.addChecker<ModuleChecker>("nasac.ModuleChecker","No desc");
    });
    AnalyzerOptions->CheckersControlList.push_back({"nasac.ModuleChecker", true});
  }

  if (!NoInitInNeedCheck) {
    AnalysisConsumer->AddCheckerRegistrationFn([] (CheckerRegistry& Registry) {
      Registry.addChecker<InitInNeedChecker>("nasac.InitInNeedChecker", "No desc");
    });
    AnalyzerOptionsRef AnalyzerOptions = Compiler.getAnalyzerOpts();
    AnalyzerOptions->CheckersControlList.push_back({"nasac.InitInNeedChecker", true});
  }

  if (!NoErrorHandlingCheck) {
    AnalysisConsumer->AddCheckerRegistrationFn([] (CheckerRegistry& Registry) {
      Registry.addChecker<ErrorHandlingChecker>("nasac.ErrorHandlingChecker", "No desc");
    });
    AnalyzerOptionsRef AnalyzerOptions = Compiler.getAnalyzerOpts();
    AnalyzerOptions->CheckersControlList.push_back({"nasac.ErrorHandlingChecker", true});
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
	  "-fparse-all-comments", ArgumentInsertPosition::END));
    Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
	  "-w", ArgumentInsertPosition::END));

  std::unique_ptr<FrontendActionFactory> FrontendFactory;
  FrontendFactory = newFrontendActionFactory<CodingSpecCheckAction>();
  Tool.run(FrontendFactory.get());

  high_resolution_clock::time_point End = high_resolution_clock::now();

  if (TurnOnTiming) {
    double TimeSpan = duration_cast<duration<double>>(End - Begin).count();
    TimeSpans.push_back(TimeSpan);
    OutputTimingInfo();
  }
  
  llvm::Error FinalizationError = ToolFinalization();
  if (FinalizationError) {
    llvm::report_fatal_error("[CodingSpecChecker]:" +
      llvm::toString(std::move(FinalizationError)));
  }
}
