#include "ClassCheck.h"
#include <string>
#include <iostream>
#include <unordered_set>

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang;
using namespace ento;
using namespace clang::ast_matchers;
std::unordered_set<std::string> friend_set;

#define LOCATION(D) \
    auto& sm = D->getASTContext().getSourceManager();\
    PathDiagnosticLocation location(D,sm);

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
static llvm::StringRef category = "style-suggestion";

///\brief judge if the method is a get/set method
bool isGetorSet(std::string declName){
    if(declName.length() < 3) return false;
    if((declName[0] == 'g'||declName[0]=='s')
        &&declName[1]=='e'&& declName[2]=='t'){
            return true;
    }
    return false;
}
///\brief judge methods for specific Field
///
///isSetforField judge the existance of the set method for a Field
///isGetforField judge the existance of the get method for a Field
bool isSetforField(std::string declName, std::string fieldName){
    if(declName.length() < 3) return false;
    if(declName[0]=='s' && declName[1]=='e' && declName[2]=='t'
        && declName[3]=='_'){
            if (declName.length() - 4 != fieldName.length()){
                return false;
            }
            for (int i=0; i<fieldName.length(); i++){
                if (fieldName[i] != declName[i+4]){
                    return false;
                }
            }
            return true;
    }
    return false;
}

bool isGetforField(std::string declName, std::string fieldName){
    if(declName.length() < 3) return false;
    if(declName[0]=='g' && declName[1]=='e' && declName[2]=='t'
        && declName[3]=='_'){
            if (declName.length() - 4 != fieldName.length()){
                return false;
            }
            for (int i=0; i<fieldName.length(); i++){
                if (fieldName[i] != declName[i+4]){
                    return false;
                }
            }
            return true;
    }
    return false;
}

///\brief retrieve the number of the function parameters.
int parmSize(const FunctionDecl *D){
    llvm::ArrayRef<ParmVarDecl*> parameters = D->parameters();
    return parameters.size();
}
bool findCopyAssignment(const CXXRecordDecl *D){
    auto method_begin = D->method_begin();
    auto method_end = D->method_end();
    while(method_begin != method_end){
        if(method_begin->isCopyAssignmentOperator()) return true;
        method_begin++;
    }
    return false;
}
bool findMoveassignment(const CXXRecordDecl *D){
    auto method_begin = D->method_begin();
    auto method_end = D->method_end();
    while(method_begin != method_end){
        if(method_begin->isMoveAssignmentOperator()) return true;
        method_begin++;
    }
    return false;
}
bool endWith(std::string& str1,std::string str2){
    int index1 = str1.size()-1;
    int index2 = str2.size()-1;
    if(index1 < index2) return false;
    while(index2 >= 0 && str1[index1--] == str2[index2]--);
    if(index2 < 0) return true;
    return false;
}

///\brief This is the main entry of all class check
void ClassChecker::checkASTDecl(const CXXRecordDecl *D, AnalysisManager &Mgr, BugReporter &BR)const{
    BEGIN("checkASTDecl");
    COUT(D->getNameAsString());
    // std::cout<< D->getOwningModuleID() << std::endl;
    if(D->isStruct()){
        structsVsClassCheck(D,Mgr,BR);
        return;
    }
    auto bases_begin = D->bases_begin();
    auto bases_end = D->bases_end();
    while(bases_begin != bases_end){
        baseClassCheck(D, &*bases_begin,Mgr,BR);
        bases_begin++;
    }

    auto ctor_begin = D->ctor_begin();
    auto ctor_end = D->ctor_end();
    while(ctor_begin != ctor_end){
        auto pCtor = *ctor_begin;
        cxxConstructorDeclCheck(&*pCtor,Mgr,BR);
        ctor_begin++;
    }

    auto method_begin = D->method_begin();
    auto method_end = D->method_end();
    while(method_begin != method_end){
        auto pMethod = *method_begin;
        cxxMethodDeclCheck(&*pMethod, Mgr,BR);
        method_begin++;
    }

    auto friend_begin = D->friend_begin();
    auto friend_end = D->friend_end();
    while(friend_begin != friend_end){
        auto pFriend = *friend_begin;
        friendCheck(&*pFriend,Mgr, BR);
        friend_begin++;
    }

    auto field_begin = D->field_begin();
    auto field_end = D->field_end();
    while(field_begin != field_end){
        auto pField = *field_begin;
        fieldDeclCheck(&*pField, Mgr,BR);
        field_begin++;
    }
    declarationOrder(D, Mgr, BR);
    cxxDestructorDeclCheck(D->getDestructor(),Mgr,BR);
    //multiInheritanceCheck(D,Mgr,BR);
    END("checlASTDecl");
}

void ClassChecker::checkEndAnalysis(ExplodedGraph &G, BugReporter &BR, ExprEngine &Eng)const{

}

void ClassChecker::cxxConstructorDeclCheck(const CXXConstructorDecl * D, AnalysisManager &Mgr, BugReporter &BR)const{
    implicitConversionsCheck_Ctor(D,Mgr,BR);
    copyableAndMovableTypeCheck(D,Mgr,BR);
}
void ClassChecker::cxxDestructorDeclCheck(const CXXDestructorDecl *D, AnalysisManager &Mgr, BugReporter &BR)const{

}
void ClassChecker::cxxMethodDeclCheck(const CXXMethodDecl *D, AnalysisManager &Mgr, BugReporter &BR)const{
    implicitConversionsCheck_Operator(D,Mgr,BR);
    virtualspecifierCheck(D, Mgr, BR);
    operatorOverloadingCheck(D, Mgr, BR);
    if(friend_set.count(D->getNameAsString()) != 0){
        friend_set.erase((std::string)D->getNameAsString());
    }
}
void ClassChecker::baseClassCheck(const CXXRecordDecl *D, const CXXBaseSpecifier *B, AnalysisManager &Mgr, BugReporter &BR)const{
    inheritanceCheck(D, B, Mgr, BR);
}
void ClassChecker::friendCheck(const FriendDecl *D, AnalysisManager &Mgr, BugReporter &BR)const{
    COUT("friend:");
    auto ND = D->getFriendDecl();
    if(ND){
        //std::cout<< ND->getNameAsString() << std::endl;
        friend_set.insert(ND->getNameAsString());
    }
    // std::cout<< D->get << std::endl;

}
void ClassChecker::fieldDeclCheck(const FieldDecl *D, AnalysisManager &Mgr, BugReporter &BR)const{
    accessControl(D, Mgr, BR);
}
///\brief Avoid virtual method calls in constructors, and avoid initialization that can fail if you can't signal an error.
///
///\TODO: How to check the initialization that can fail if you can't signal an error
///
///checking this from ast decl is hard. So I changed the strategy. We can check this
///by checkPreCall. If the virtual function is called, we check whether the caller is
///a constructor.
void ClassChecker::workInCtorCheck(const CallEvent &call, CheckerContext &C)const{
    BEGIN("workInCtorCheck");
    const Decl* D = call.getDecl();
    auto FD = static_cast<const FunctionDecl*>(D);
    if(!FD) return;
    auto MD = static_cast<const CXXMethodDecl*>(FD);
    if(!MD) return;
    if(!MD->isVirtual()) return;
    const auto RD = MD->getParent();
    auto ctor_begin = RD->ctor_begin();
    auto ctor_end = RD->ctor_end();
    auto Callee = call.getCalleeIdentifier();
    auto SF = C.getStackFrame();
    auto SFD = SF->getDecl();
    if(!SFD)  return;
    auto SFFuncDecl = static_cast<const FunctionDecl*>(SFD);
    if(!SFFuncDecl) return;
    auto SFMethodDecl = static_cast<const CXXMethodDecl*>(SFFuncDecl);
    if(!SFMethodDecl) return;
    auto SFCtorDecl = static_cast<const CXXConstructorDecl*>(SFMethodDecl);
    if(!SFCtorDecl) return;

    LOCATION(SFCtorDecl);
    auto& BR = C.getBugReporter();
    BR.EmitBasicReport(SFCtorDecl,this,
        "ctor call virtual bug",
        category,
        "the constructor should not call virtual member function",
        location);
    END("workInCtorCheck");
}

void ClassChecker::implicitConversionsCheck_Ctor(const CXXConstructorDecl* D, AnalysisManager &Mgr, BugReporter &BR)const{
    if(D->isMoveConstructor()) return;
    if(D->isCopyConstructor()) return;
    if(D->isExplicit()) return;
    if(parmSize(D) == 1){
        LOCATION(D);
        BR.EmitBasicReport(D,this,
        "ImplicitConversions",
        category,
        "single parameter constructor may cause implicit conversion",
        location);
    }

}
void ClassChecker::implicitConversionsCheck_Operator(
    const CXXMethodDecl *D, AnalysisManager &Mgr, BugReporter &BR)const{

    auto ConvD = static_cast<const CXXConversionDecl*>(D);
    if(!ConvD) return;
    auto CtorD = static_cast<const CXXConstructorDecl*>(D);
    if(CtorD) return;
    if(ConvD->isExplicit()) return;
    LOCATION(D);
    BR.EmitBasicReport(D,this,
        "ImplicitConversions",
        category,
        "Conversion function should be explicit",
        location);
}

///\brief A class's public API should make explicit whether the class is copyable, move-only, or neither copyable nor movable.
///
///This rule is unclear, which makes it hard to check.
///For now, I do these jobs:
///     1. check whether the copy(or move) ctor and copy(or move) assignment is together or not.
///     2. if the copy/move ctor is not declared explicitly, whether it is marked as 'delete'
void ClassChecker::copyableAndMovableTypeCheck(const CXXConstructorDecl *D, AnalysisManager &Mgr, BugReporter &BR)const{
    if(!D->isCopyOrMoveConstructor()) return;
    auto record = D->getParent();
    LOCATION(record);
    if(!D->isUserProvided()&&!D->isDeleted()){
        BR.EmitBasicReport(record, this,
            "Copy/Move Type Error",
            category,
            "The copy/move ctor should be provided or marked delete",
            location);
    }
    if(D->isCopyConstructor()){
        if(findCopyAssignment(record))
            return;
        BR.EmitBasicReport(record, this,
            "CopyAssignment Lack",
            category,
            "Copy Ctor should have responsive copy assignment operator",
            location);
        return;
    }
    /// move constructor
    if(findMoveassignment(record))
        return;
    BR.EmitBasicReport(record, this,
        "Move Assignment Lack",
        category,
        "Move Ctor should have responsive move assignment operator",
        location);
}

void ClassChecker::structsVsClassCheck(const CXXRecordDecl *D, AnalysisManager &Mgr, BugReporter &BR) const{
    if(!D->isStruct()) return;
    if(D->isLambda()) return;
    auto method_begin = D->method_begin();
    auto method_end = D->method_end();
    bool should_be_class = false;
    LOCATION(D);
    CXXMethodDecl *errMethod = nullptr;
    while(method_begin != method_end){
        auto method = *method_begin;
        method_begin++;
        if(method->isStatic()) continue;
        COUT(method->getNameAsString());
        if(isGetorSet(method->getNameAsString())){
            BR.EmitBasicReport(method,this,
            "unnessesery struct access/setting function",
            category,
            "In struct, the functionality of access/setting of data member is unnnessesary",
            location);
            continue;
        }
        if(parmSize(method)>= 1){
            should_be_class = true;
            errMethod = method;
            break;
        }

    }
    if(should_be_class){
        BR.EmitBasicReport(errMethod,this,
        "struct functionality",
        category,
        "struct should be passive object and have few functionality",
        location);
    }
}

///\brief all base classes should be inherited in a public access way
void ClassChecker::inheritanceCheck(const CXXRecordDecl *D, const CXXBaseSpecifier *B, AnalysisManager &Mgr, BugReporter &BR)const{
    if (B->getAccessSpecifier() != AS_public){
        LOCATION(D);
        BR.EmitBasicReport(D,this,
        "non_public inheritance",
        category,
        "the access specifier can only be public.  advice: using composition rather than inheritance",
        location);
    }
    /// there is something that need to check in methoddecl
}

///\brief every virtual function inherit from base class should be marked explicitly
void ClassChecker::virtualspecifierCheck(const CXXMethodDecl *D, AnalysisManager &Mgr, BugReporter &BR)const{
    if(D->isVirtual()){
        if (!D->isVirtualAsWritten()){
            LOCATION(D);
            BR.EmitBasicReport(D,this,
            "implicit virtual function",
            category,
            "In order to recognize virtual funciton, every method that inherits from base class's virtual method should use virtual specifier explicitly.",
            location);
        }
    }
}

///\brief overloading operator is not allowed
///
///hard to check the time consuming of overloading operator
///also hard to resort to the origin code
void ClassChecker::operatorOverloadingCheck(const CXXMethodDecl *D, AnalysisManager &Mgr, BugReporter &BR)const{
    if (D->isOverloadedOperator()){
        if (D->getOverloadedOperator() == OO_EqualEqual){
            LOCATION(D);
            BR.EmitBasicReport(D,this,
            "necessary use of operator overload",
            category,
            "not recommend to overload /'==/', if necessary, please provide documentation.",
            location);
        }else{
            LOCATION(D);
            BR.EmitBasicReport(D,this,
            "operator overload",
            category,
            "not recommend to overload operator, because of the complexity of check of time and source code.",
            location);
        }
    }
}
///\brief data field must declare as private
///
///every data field also needs a pair of get/set methods
void ClassChecker::accessControl(const FieldDecl *D, AnalysisManager &Mgr, BugReporter &BR)const{
    if(D->getType()->isEnumeralType()) return;
    // auto type = D->getType();
    // if(static_cast<QualType>(type)->isConstant(D->getASTContext())) return;
    if (D->getAccess() != AS_private){
        LOCATION(D);
        BR.EmitBasicReport(D,this,
        "FieldAccessControl",
        category,
        "All data field should declare as private with get/set method.",
        location);
    }else{
        bool has_get = false;
        bool has_set = false;
        auto RD = static_cast<const CXXRecordDecl *>(D->getParent());
        auto MD_begin = RD->method_begin();
        auto MD_end = RD->method_end();
        while(MD_begin!=MD_end){
            if(isSetforField((*MD_begin)->getNameAsString(), D->getNameAsString())){
                has_set = true;
            }
            if(isGetforField((*MD_begin)->getNameAsString(), D->getNameAsString())){
                has_get = true;
            }
            MD_begin++;
        }
        if (!has_get){
            LOCATION(D);
            BR.EmitBasicReport(D,this,
            "FieldAccessControlFunction",
            category,
            "Every data field should have a get method. eg: get_fieldname()",
            location);
        }
        if (!has_set){
            LOCATION(D);
            BR.EmitBasicReport(D,this,
            "FieldAccessControlFunction",
            category,
            "Every data field should have a set method. eg: set_fieldname()",
            location);
        }
    }
}

///\brief: check the regular way to order the declaration of members in a class
///
///Main block divide:
///     1. public
///     2. protected
///     3. private
///order in a block:
///     1. typedef/enum
///     2. const
///     3. constructor
///     4. destructor
///     5. member function(method)
///     6. data field
void ClassChecker::declarationOrder(const CXXRecordDecl *D, AnalysisManager &Mgr, BugReporter &BR)const{
    auto decl_begin = D->decls_begin();
    auto decl_end = D->decls_end();
    ///\access_flag: public:1 protected:2 private:3
    int access_flag = 0;
    ///\type_flag: typedef/enum:1 const:2 constructor:3 destructor:4  memberfunction:5 datafield:6
    int type_flag = 0;
    int public_count = 0;
    int protected_count = 0;
    int private_count = 0;
    while(decl_begin != decl_end){
        auto decl = *decl_begin;
        std::string kind = decl->getDeclKindName();
        if(kind == "AccessSpec"){
            if(decl->getAccess()==AS_public){
                if(access_flag <= 1){
                    access_flag = 1;
                    type_flag = 0;
                }else{
                    LOCATION(decl);
                    BR.EmitBasicReport(decl,this,
                    "DeclarationOrder",
                    category,
                    "members should obey the order like public, protected, private",
                    location);
                }
            }
            if(decl->getAccess()==AS_protected){
                if(access_flag <= 2){
                    access_flag = 2;
                    type_flag = 0;
                }else{
                    LOCATION(decl);
                    BR.EmitBasicReport(decl,this,
                    "DeclarationOrder",
                    category,
                    "members should obey the order like public, protected, private",
                    location);
                }
            }
            if(decl->getAccess()==AS_private){
                if(access_flag <= 3){
                    access_flag = 3;
                    type_flag = 0;
                }else{
                    LOCATION(decl);
                    BR.EmitBasicReport(decl,this,
                    "DeclarationOrder",
                    category,
                    "members should obey the order like public, protected, private",
                    location);
                }
            }
        }else{
            if(kind == "Typedef" || kind == "Enum"){
                if (type_flag <= 1){
                    type_flag = 1;
                }else{
                    LOCATION(decl);
                    BR.EmitBasicReport(decl,this,
                    "DeclarationOrder",
                    category,
                    "members in a access block should obey the order: typedef/enum -> const -> constructor -> destructor -> member function -> data field",
                    location);
                }
            }
            if(kind == "CXXConstructor"){
                /// blong to constructor
                if (type_flag <= 3){
                    type_flag = 3;
                }else{
                    LOCATION(decl);
                    BR.EmitBasicReport(decl,this,
                    "DeclarationOrder",
                    category,
                    "members in a access block should obey the order: typedef/enum -> const -> constructor -> destructor -> member function -> data field",
                    location);
                }
            }
            if(kind == "CXXDestructor"){
                /// blong to destructor
                if (type_flag <= 4){
                    type_flag = 4;
                }else{
                    LOCATION(decl);
                    BR.EmitBasicReport(decl,this,
                    "DeclarationOrder",
                    category,
                    "members in a access block should obey the order: typedef/enum -> const -> constructor -> destructor -> member function -> data field",
                    location);
                }
            }
            if(kind == "CXXMethod"){
                /// blong to Method
                /// blong to normal member function
                if (type_flag <= 5){
                    type_flag = 5;
                }else{
                    LOCATION(decl);
                    BR.EmitBasicReport(decl,this,
                    "DeclarationOrder",
                    category,
                    "members in a access block should obey the order: typedef/enum -> const -> constructor -> destructor -> member function -> data field",
                    location);
                }
            }
            if(kind == "Field"){
                /// blong to data Field
                if (type_flag <= 6){
                    type_flag = 6;
                }else{
                    LOCATION(decl);
                    BR.EmitBasicReport(decl,this,
                    "DeclarationOrder",
                    category,
                    "members in a access block should obey the order: typedef/enum -> const -> constructor -> destructor -> member function -> data field",
                    location);
                }
            }
        }

        decl_begin++;
    }



}

//////////////////////////////////////////////////////////////////////////////

void ClassChecker::checkPreCall(const CallEvent &call, CheckerContext &C)const{
    BEGIN("checkPreCall");
    workInCtorCheck(call,C);
    END("checkPreCall");

}
#undef BEGIN
#undef END
#undef COUT
#undef LOCATION
