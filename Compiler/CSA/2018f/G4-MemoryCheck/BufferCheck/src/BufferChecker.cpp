/// @version 0.1.0
/// @author Li Nan
////////////////////////////////////////////////////////////////////////////////

#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerHelpers.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "StaticAnalyzer/Checkers/ClangSACheckers.h"
//#include "clang/StaticAnalyzer/Core/CheckerRegistry.h"

using namespace clang;
using namespace clang::ento;

class BufferChecker:public Checker<check::PreCall>
{
public:
	BufferChecker();
	mutable std::unique_ptr<BugType> BT;
	void checkPreCall(const CallEvent &Call,CheckerContext &C) const;
	~BufferChecker();
};

BufferChecker::BufferChecker()
{}

BufferChecker::~BufferChecker()
{}

void BufferChecker::checkPreCall(const CallEvent &Call,CheckerContext &C) const
{
	if(const IdentifierInfo *II = Call.getCalleeIdentifier())
	{
		if(II->isStr("strcpy") || II->isStr("memcpy"))
		{
			SVal DstSVal = Call.getArgSVal(0);
			SVal SrcSVal = Call.getArgSVal(1);								//获取函数参数src和dst

			const MemRegion *dstMemRegion = DstSVal.getAsRegion();	
			const MemRegion *srcMemRegion = SrcSVal.getAsRegion();
			if(!dstMemRegion || !srcMemRegion)return;
			
			const MemRegion *dstBaseR = dstMemRegion->getBaseRegion();
  			const MemRegion *srcBaseR = srcMemRegion->getBaseRegion();
			if(!dstBaseR || !srcBaseR)return;
  			
			const SubRegion *dstSubRegion = dyn_cast_or_null<SubRegion>(dstBaseR);
			const SubRegion *srcSubRegion = dyn_cast_or_null<SubRegion>(srcBaseR);
  			if(!dstSubRegion || !srcSubRegion)return;

  			SValBuilder &svalBuilder = C.getSValBuilder();
  			SVal dstExtent = dstSubRegion->getExtent(svalBuilder);
			SVal srcExtent = srcSubRegion->getExtent(svalBuilder);

  			ProgramStateRef state = C.getState();
  			Optional<DefinedSVal> dstSizeDSVal = dstExtent.getAs<DefinedSVal>();
			Optional<DefinedSVal> srcSizeDSVal = srcExtent.getAs<DefinedSVal>();		//获取src和dst的大小
			if(!dstSizeDSVal || !srcSizeDSVal)return;

			SVal dstLTSrc = svalBuilder.evalBinOp(state,BO_LT,*dstSizeDSVal,*srcSizeDSVal,svalBuilder.getConditionType());
			Optional<DefinedSVal> dstLTSrcDSVal = dstLTSrc.getAs<DefinedSVal>();		//符号执行，比较src和dst的大小

			if(!dstLTSrcDSVal)return;
			ConstraintManager &CM = C.getConstraintManager();
			ProgramStateRef stateDstLTSrc = CM.assume(state,*dstLTSrcDSVal,true);
			if(stateDstLTSrc)															//如果dst小于src，则报告一个bug
			{
				BT.reset(new BugType(this,"DstBufferTooSmall","MemoryChecker"));
				ExplodedNode *N = C.generateErrorNode();
				auto report = llvm::make_unique<BugReport>(*BT,BT->getName(),N);
				C.emitReport(std::move(report));
			}

			if(II->isStr("memcpy"))					//如果是memcpy函数，则还需考虑其第三个参数，src和dst均不能小于它
			{
				SVal size = Call.getArgSVal(2);
				Optional<DefinedSVal> sizeDSVal = size.getAs<DefinedSVal>();
				SVal dstLTSize = svalBuilder.evalBinOp(state,BO_LT,*dstSizeDSVal,*sizeDSVal,svalBuilder.getConditionType());
				SVal srcLTSize = svalBuilder.evalBinOp(state,BO_LT,*srcSizeDSVal,*sizeDSVal,svalBuilder.getConditionType());
				Optional<DefinedSVal> dstLTSizeDSVal = dstLTSize.getAs<DefinedSVal>();
				Optional<DefinedSVal> srcLTSizeDSVal = srcLTSize.getAs<DefinedSVal>();
				if(dstLTSizeDSVal)
				{
					ProgramStateRef stateDstLTSize = CM.assume(state,*dstLTSizeDSVal,true);
					if(stateDstLTSize)
					{
						BT.reset(new BugType(this,"SizeLargerThanDstBuffer","MemoryChecker"));
						ExplodedNode *N = C.generateErrorNode();
						auto report = llvm::make_unique<BugReport>(*BT,BT->getName(),N);
						C.emitReport(std::move(report));
					}
				}
				if(srcLTSizeDSVal)
				{
					ProgramStateRef stateSrcLTSize = CM.assume(state,*srcLTSizeDSVal,true);
					if(stateSrcLTSize)
					{
						BT.reset(new BugType(this,"SizeLargerThanSrcBuffer","MemoryChecker"));
						ExplodedNode *N = C.generateErrorNode();
						auto report = llvm::make_unique<BugReport>(*BT,BT->getName(),N);
						C.emitReport(std::move(report));
					}
				}
			}
		}
	}
}

void ento::registerBufferChecker(CheckerManager &mgr) 			//注册该checker
{
  mgr.registerChecker<BufferChecker>();
}