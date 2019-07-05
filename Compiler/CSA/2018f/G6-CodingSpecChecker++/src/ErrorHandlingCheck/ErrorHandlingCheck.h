////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2018, University of Science and Techonolgy of China
/// All rights reserved.
///
/// @file ErrorHandlingCheck.h
/// @brief Main entry of this anlysis tool, which parses arguments and dispatches 
/// to corresponding FrontendAction instances
///
/// @version 0.1.0
/// @author Yuxiang Zhang, <a href="linkURL">zyx504@mail.ustc.edu.cn</a> 
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include <unordered_set>

namespace clang {
  namespace ento {

    class ErrorHandlingChecker: public Checker<check::PostCall,
                                               check::BranchCondition> {
      static std::unordered_set<std::string> fName;

    public:

      void checkPostCall(const CallEvent &Call, CheckerContext &C) const;

      void checkBranchCondition(const Stmt *Condition, CheckerContext &C) const;

      static void StaticInit(const std::string &dict_dir);
    };

  } // end namespace ento
} // end namespace clang
