////////////////////////////////////////////////////////////////////////////////
/// Copyright (c) 2018, University of Science and Techonolgy of China
/// All rights reserved.
///
/// @file InitInNeedCheck.cpp
/// @brief Main entry of this anlysis tool, which parses arguments and dispatches
/// to corresponding FrontendAction instances
///
/// @version 0.1.0
/// @author Yuxiang Zhang (Leader), <a href="linkURL">zyx504@mail.ustc.edu.cn</a> 
/// @author Shengliang Deng, <a href="linkURL">dengsl@mail.ustc.edu.cn</a> 
/// @author Yu Zhang (Mentor), <a href="linkURL">yuzhang@ustc.edu.cn</a> 
////////////////////////////////////////////////////////////////////////////////

#include "InitInNeedCheck.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporterVisitors.h"

using namespace clang;
using namespace clang::ento;

struct PrevAccState
{
  private:
	enum Kind
	{
		READ,
		WRITE,
		WriteBeforeBranch
	} K;
	const Stmt *PrevAccStmt;
	const Decl *PrevAccDecl;

	PrevAccState(Kind InitK, const Stmt *S) : K(InitK), PrevAccStmt(S), PrevAccDecl(nullptr) {}
	PrevAccState(Kind InitK, const Decl *D) : K(InitK), PrevAccStmt(nullptr), PrevAccDecl(D) {}

  public:
	bool operator==(const PrevAccState &S) const { 
		return S.K == K && S.PrevAccStmt == PrevAccStmt && S.PrevAccDecl == PrevAccDecl;
	}
	void Profile(llvm::FoldingSetNodeID &ID) const { 
		ID.AddInteger(K); 
	}

	static PrevAccState getRead(const Stmt *S) { return PrevAccState(READ, S); }
	static PrevAccState getWrite(const Stmt *S) { return PrevAccState(WRITE, S); }
	static PrevAccState getWrite(const Decl *D) { return PrevAccState(WRITE, D); }
	static PrevAccState getWriteBeforeBranch(PrevAccState PrevAcc) { 
		PrevAccState Acc = PrevAcc;
		Acc.K = WriteBeforeBranch;
		return Acc;
	}

	bool isRead() const { return K == READ; }
	bool isWrite() const { return K == WRITE; }
	const Stmt *getStmt() const { return PrevAccStmt; }
	const Decl *getDecl() const { return PrevAccDecl; }
};

REGISTER_MAP_WITH_PROGRAMSTATE(AccessState, const MemRegion *, PrevAccState)

InitInNeedChecker::InitInNeedChecker()
{
	// Initialize the bug types.
	RedundantWriteBugType.reset(
		new BugType(this, "reduntant write", "NASAC 2018 Coding Specification Check"));
}

void InitInNeedChecker::reportReduntantWrite(
	ExplodedNode *ErrNode, CheckerContext &Ctx) const
{
	std::unique_ptr<BugReport> R = llvm::make_unique<BugReport>(
		*RedundantWriteBugType, RedundantWriteBugType->getName(), ErrNode);
	Ctx.emitReport(std::move(R));
}

void InitInNeedChecker::checkLocation(
	SVal Loc, bool IsLoad, const Stmt *S, CheckerContext &Ctx) const
{

	ProgramStateRef State = Ctx.getState();

	const MemRegion *Region = Loc.getAsRegion();
	if (Region && IsLoad)
	{
		State = State->set<AccessState>(Region, PrevAccState::getRead(S));
	}

	if (State != Ctx.getState())
	{
		Ctx.addTransition(State);
	}
}

static const MemRegion *GetSuperMemReigon(
	const MemRegion *MR, ProgramStateRef &State)
{
	while (const SubRegion *SR = dyn_cast<SubRegion>(MR))
	{
		MR = SR->getSuperRegion();
		if (State->contains<AccessState>(MR))
		{
			break;
		}
	}
	return MR;
}

/// FIXME: I can not ensure this condition covers all cases of output parameter
static bool IsOutputParameter(const FunctionDecl *FD, int I)
{
	const ParmVarDecl *PD = FD->getParamDecl(I);
	if (!PD)
	{
		return false;
	}
	const QualType QT = PD->getType();
	const Type *RT = QT.getTypePtr();
	return RT && RT->isPointerType() && !RT->getPointeeType().isConstQualified();
}

bool InitInNeedChecker::evalCall(
	const CallExpr *CE, CheckerContext &Ctx) const
{

	ProgramStateRef State = Ctx.getState();
	SValBuilder &SVB = Ctx.getSValBuilder();
	const LocationContext *LC = Ctx.getLocationContext();
	const FunctionDecl *FD = CE->getDirectCallee();

	if (!FD)
	{
		return false;
	}

	for (int I = 0, N = FD->getNumParams(); I != N; I++)
	{
		const Expr *Arg = CE->getArg(I);
		const SVal Val = Ctx.getSVal(Arg);
		const MemRegion *Region = Val.getAsRegion();

		if (!Region)
		{
			continue;
		}
		
		const TypedValueRegion *BaseRegion = 
			dyn_cast<TypedValueRegion>(Region->getBaseRegion());
		if (!BaseRegion) {
			continue;
		}
		
		//const MemRegion *SuperRegion = GetSuperMemReigon(BaseRegion, State);

		//// 如果写的内存区域是全局内存区域
		//if (SuperRegion->getKind() >= MemRegion::BEGIN_GLOBAL_MEMSPACES
		//	&& SuperRegion->getKind() < MemRegion::END_GLOBAL_MEMSPACES) {
		//	// 如果之前没有访问过，那么检查该位置绑定的右值
		//	if (!State->contains<AccessState>(BaseRegion)) {
		//		Loc LV = SVB.makeLoc(BaseRegion);
		//		SVal Val = Ctx.getStoreManager().getBinding(State->getStore(), LV);
		//		if (!Val.isUnknownOrUndef()) {
		//			State = State->set<AccessState>(BaseRegion, PrevAccState::getWrite(nullptr));
		//		}

		//	}
		//}

		if (IsOutputParameter(FD, I))
		{
			const PrevAccState *PrevAccess = State->get<AccessState>(BaseRegion);

			if (PrevAccess && PrevAccess->isWrite())
			{

				// 非全局变量的冗余写
				if (const Stmt *PrevWrite = PrevAccess->getStmt()) {
					{
						DiagnosticsEngine &DE = Ctx.getBugReporter().getDiagnostic();
						unsigned ID = DE.getCustomDiagID(DiagnosticsEngine::Warning,
							"reduntant write");
						DE.Report(PrevWrite->getBeginLoc(), ID);
					}
					{
						DiagnosticsEngine &DE = Ctx.getBugReporter().getDiagnostic();
						unsigned ID = DE.getCustomDiagID(DiagnosticsEngine::Note,
							"following write");
						DE.Report(Arg->getBeginLoc(), ID);
					}
				}
				// 全局变量的冗余写
				
				if (const Decl *PrevWrite = PrevAccess->getDecl()) {
					{
						DiagnosticsEngine &DE = Ctx.getBugReporter().getDiagnostic();
						unsigned ID = DE.getCustomDiagID(DiagnosticsEngine::Warning,
							"Maybe write a global variable with reduntant initilization");
						DE.Report(PrevWrite->getBeginLoc(), ID);
					}
					{
						DiagnosticsEngine &DE = Ctx.getBugReporter().getDiagnostic();
						unsigned ID = DE.getCustomDiagID(DiagnosticsEngine::Note,
							"following write");
						DE.Report(Arg->getBeginLoc(), ID);
					}
				}
			}
			State = State->set<AccessState>(Region, PrevAccState::getWrite(Arg));
		}
		else
		{
			State = State->set<AccessState>(Region, PrevAccState::getRead(Arg));
		}
	}

	QualType RetType = CE->getCallReturnType(Ctx.getASTContext());
	SVal RetConjured = SVB.conjureSymbolVal(CE, LC, RetType, Ctx.blockCount());
	State = State->BindExpr(CE, LC, RetConjured);

	if (State != Ctx.getState())
	{
		Ctx.addTransition(State);
	}

	return true;
}

void InitInNeedChecker::checkBind(
	SVal Loca, SVal Val, const Stmt *S, CheckerContext &Ctx) const
{
	ProgramStateRef State = Ctx.getState();
	const MemRegion *Region = Loca.getAsRegion();
	if (!Region)
	{
		return;
	}

	//const MemRegion *SuperRegion = GetSuperMemReigon(Region, State);

	//// 如果写的内存区域是全局内存区域
	//if (SuperRegion->getKind() >= MemRegion::BEGIN_GLOBAL_MEMSPACES 
	//	&& SuperRegion->getKind() < MemRegion::END_GLOBAL_MEMSPACES) {
	//	// 如果之前没有访问过，那么检查该位置绑定的右值
	//	if (!State->contains<AccessState>(Region)) {
	//		SVal InitVal = Ctx.getStoreManager().
	//			getBinding(State->getStore(), Loca.castAs<Loc>());
	//		InitVal.dump();
	//		// 如果该全局变量没有初始化，那么该位置绑定的值为Unknown
	//		// 否则认为该全局变量在声明时已经初始化
	//		if (!InitVal.isUnknownOrUndef()) {
	//			State = State->set<AccessState>(Region, PrevAccState::getWrite(nullptr));
	//		}
	//	}
	//}

	// 针对不是变量声明的语句，需要检查是否存在冗余写操作
	if (!isa<DeclStmt>(S))
	{
		const PrevAccState *PrevAccess = State->get<AccessState>(Region);
		if (PrevAccess && PrevAccess->isWrite())
		{
			// 非全局变量的冗余写
			if (const Stmt *PrevWrite = PrevAccess->getStmt()) {
				{
					DiagnosticsEngine &DE = Ctx.getBugReporter().getDiagnostic();
					unsigned ID = DE.getCustomDiagID(DiagnosticsEngine::Warning, 
						"reduntant write");
					DE.Report(PrevWrite->getBeginLoc(), ID);
				}
				{
					DiagnosticsEngine &DE = Ctx.getBugReporter().getDiagnostic();
					unsigned ID = DE.getCustomDiagID(DiagnosticsEngine::Note,
						"following write");
					DE.Report(S->getBeginLoc(), ID);
				}
			}
			
			if (const Decl *PrevWrite = PrevAccess->getDecl()){
				{
					DiagnosticsEngine &DE = Ctx.getBugReporter().getDiagnostic();
					unsigned ID = DE.getCustomDiagID(DiagnosticsEngine::Warning,
						"maybe write a global variable with reduntant initilization");
					DE.Report(PrevWrite->getBeginLoc(), ID);
				}
				{
					DiagnosticsEngine &DE = Ctx.getBugReporter().getDiagnostic();
					unsigned ID = DE.getCustomDiagID(DiagnosticsEngine::Note,
						"following write");
					DE.Report(S->getBeginLoc(), ID);
				}
			}
		}
	}

	// 更新当前内存区的访问状态
	State = State->set<AccessState>(Region, PrevAccState::getWrite(S));

	if (State != Ctx.getState())
	{
		Ctx.addTransition(State);
	}
}

void InitInNeedChecker::checkBranchCondition(
	const Stmt *Condition, CheckerContext &Ctx) const
{
	ProgramStateRef State = Ctx.getState();

	auto AccessStateMap = State->get<AccessState>();
	for (auto I = AccessStateMap.begin(), E = AccessStateMap.end(); I != E; I++)
	{
		State = State->remove<AccessState>(I->first);
	}

	if (State != Ctx.getState())
	{
		Ctx.addTransition(State);
	}
}

void InitInNeedChecker::checkASTDecl(
	const VarDecl *D, AnalysisManager &Mgr, BugReporter &BR) const {
	if (D->hasGlobalStorage() && D->hasInit()) {
		GlobalVarDecl.insert(D);
	}
}

void InitInNeedChecker::checkBeginFunction(CheckerContext &Ctx) const {
	ProgramStateRef State = Ctx.getState();
	auto LC = Ctx.getLocationContext();
	for (auto Global : GlobalVarDecl) {
		auto Region = State->getLValue(Global, LC).getAsRegion();
		State = State->set<AccessState>(Region, PrevAccState::getWrite(Global));
	}
	if (State != Ctx.getState()) {
		Ctx.addTransition(State);
	}
}

void InitInNeedChecker::checkEndFunction(const ReturnStmt *R, CheckerContext &Ctx) const {
	//ProgramStateRef State = Ctx.getState();
	//auto AccStateMap = State->get<AccessState>();
	//for (auto Pair : AccStateMap) {
	//	auto PrevAcc = Pair.second;

	//	if (PrevAcc.getStmt()) {
	//		PrevAcc.getStmt()->dump();
	//	}
	//	if (PrevAcc.getDecl()) {
	//		PrevAcc.getDecl()->dump();
	//	}

	//	if (PrevAcc.isWrite()) {
	//		continue;
	//	}

	//	if (const Stmt *PrevWrite = PrevAcc.getStmt()) {
	//		DiagnosticsEngine &DE = Ctx.getBugReporter().getDiagnostic();
	//		unsigned ID = DE.getCustomDiagID(DiagnosticsEngine::Warning,
	//			"a variable is wrote but never read");
	//		DE.Report(PrevWrite->getBeginLoc(), ID);
	//	}

	//	if (const Decl *PrevWrite = PrevAcc.getDecl()) {
	//		DiagnosticsEngine &DE = Ctx.getBugReporter().getDiagnostic();
	//		unsigned ID = DE.getCustomDiagID(DiagnosticsEngine::Warning,
	//			"maybe global variable is wrote but never read");
	//		DE.Report(PrevWrite->getBeginLoc(), ID);
	//	}
	//}
}

void InitInNeedChecker::checkEndAnalysis(
	ExplodedGraph &G, BugReporter &BR, ExprEngine &Eng) const {

}
