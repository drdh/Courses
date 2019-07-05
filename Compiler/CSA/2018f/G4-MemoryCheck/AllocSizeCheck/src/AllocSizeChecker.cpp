//AllocSizeChecker.cpp
//To test when call function malloc(), judge whether the argument is zero(or a tainted value), 
//If so, report a waring 'Size is Zero'
//
//Designed by Yangziqi
//------------------------------------------------
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerHelpers.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporterVisitors.h"
#include "ClangSACheckers.h"

using namespace clang;
using namespace clang::ento;

class AllocSizeChecker : public Checker<check::PreCall> {

void reportBug(const char *Msg,
                 ProgramStateRef StateZero,
                 CheckerContext &C) const ;
public:
	mutable std::unique_ptr<BugType> BT;
    AllocSizeChecker(){};
    void checkPreCall(const CallEvent & Call , CheckerContext & C)const;

private:
};

void AllocSizeChecker::reportBug(const char *Msg,
                               ProgramStateRef StateZero,
                               CheckerContext &C) const {
if(ExplodedNode *N = C.generateErrorNode(StateZero)) {
    if (!BT)
    	BT.reset(new BuiltinBug(this, "Size error"));

    auto R = llvm::make_unique<BugReport>(*BT, Msg, N);
    bugreporter::trackNullOrUndefValue(N, bugreporter::GetDenomExpr(N), *R);
    C.emitReport(std::move(R));
}
}

void AllocSizeChecker::checkPreCall(const CallEvent & Call , CheckerContext & C)const
{
    if(const IdentifierInfo *II = Call.getCalleeIdentifier()){
        if(II->isStr("malloc")){
            auto size = Call.getArgSVal(0);		//Get Argument
            Optional<DefinedSVal> DV = size.getAs<DefinedSVal>();

            if (!DV)
                return;

            ConstraintManager &CM = C.getConstraintManager();
            ProgramStateRef stateNotZero, stateZero;
            std::tie(stateNotZero, stateZero) = CM.assumeDual(C.getState(), *DV);

            if (!stateNotZero) {	//If arg1 is zero
                assert(stateZero);
                reportBug("Size is zero", stateZero, C);
                return;
            }

            bool TaintedD = C.getState()->isTainted(*DV);
            if ((stateNotZero && stateZero && TaintedD)) {
                reportBug("Size is a tainted value, possibly zero", stateZero, C);
                return;
            }	//Arg1 is a tainted value, still warning

            C.addTransition(stateNotZero);
        }
    }
}

void ento::registerAllocSizeChecker(CheckerManager &mgr)
{//Register the checker
	mgr.registerChecker<AllocSizeChecker>();
}
