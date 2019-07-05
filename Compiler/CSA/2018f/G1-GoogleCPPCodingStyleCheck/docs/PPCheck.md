# PPChecker 设计

## 0. PPChecker实现的功能

1. 预处理宏：
   1. 不要在.h文件中定义宏
   2. 在文件内定义的宏至少在文件末尾#undef
   3. 不要只是对已经存在的宏使用#undef, 选择一个不会冲突的名称
   4. 不要使用## 处理函数， 类和变量的名字

## 1. 类设计

```c++
class PPChecker:public clang::PPCallbacks{
    public:
    	void InclusionDirective(...);
    	void EndOfMainFile();
    	void MacroDefined(...);
    	void MacroUndefined(...);
    public:
    	void checkUndef();
    	void checkDefineInHeaderFile(...);
    	void checkMultiDefine(...);
    	void checkUnrecommendedMacro(...);
    	void checkDoubleSharpInMacro(...);
    ...
```

`PPChecker`主要是继承`PPCallbacks`类，然后重载几个回调函数，在回调函数里面调用各种各种check.

### 1.1过滤非用户定义的宏

由于clang会只要在分析源码时遇到宏定义都会调用`MacroDefined`回调函数，其它的类似，这就导致遇到编译器定义的宏的时候也会调用回调函数，这样会出现很多无用的报错信息。为了避免这种情况的发生，首先需要在各回调函数前面对文件进行过滤。这就需要在`ToolDriver.cpp`里面在确定需要分析的文件后将这些文件传递到类里面然后再进行检查。为此，定义几个函数：

```c++
/// src/ToolDriver.cpp
///此函数获取需要分析的文件的绝对路径，存于全局变量SourceList里面
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
/// src/PPCheck/PPCheck.cpp
bool PPChecker::isFileInSource(string &file){
    auto source_begin = SourceList.begin();
    auto source_end = SourceList.end();
    while(source_begin != source_end){
        if(source_begin->compare(file)==0)
            return true;
        source_begin++;
    }
    return false;
}
string PPChecker::getFullPathFromLocation(const SourceLocation &location){
    auto loc_str = location.printToString(CI.getSourceManager());
    int cut_index = loc_str.find(':');
    string cut_loc = loc_str.substr(0,cut_index);
    return std::move(cut_loc);
}
```

### 1.2 检查宏是否在.h文件中定义

在`MacroDefine`回调函数被调用时检查是否是一个新文件，是新文件的化就检查是否是.h文件，如果是.h文件就报错。

```c++
void PPChecker::checkIsNewFile(string& loc){
    
    if(loc.compare(currFile)==0) return;
    checkUndef();
    checkIncludeOrder();
    macros.clear();
    //inclusionInfoSet.clear();
    isHeaderFile = false;
    currFile = loc;
    if(endWith(currFile.data(),currFile.length(),".h",2)||
        endWith(currFile.data(),currFile.length(),".hpp",4))
        isHeaderFile = true;
}
```

### 1.3 检查是否#undef定义的宏

这里主要是在分析一个新文件是将遇到的宏都存起来，此后每遇到一个`#undef`就对存储的宏进行标记，在遇到下一个新文件时检查存储的宏里面是否有没有被标记的宏。

```c++
void PPChecker::checkIsNewFile(string& loc){
    
    if(loc.compare(currFile)==0) return;
    checkUndef(); //遇到新文件后则对已存储的进行检查
    macros.clear();//然后将存储的宏清空
    ...
}
```

### 1.4 检查是否在定义宏时使用`##`

这里主要基于宏定义的`Token`的名字进行检查看看是否有`hashash`，但如果简单的判断的化字符串里面的`##`也会报错，所以需要检查紧跟的`Token`是否是`identifier`。

```c++
///主要就是遍历宏定义的Token    
while(token_begin != token_end){
        auto token = token_begin;
        token_begin++;
        if(strcmp(token->getName(),"hashhash")){
            preHashHash = true;
        }
        else if(strcmp(token->getName(),"identifier")&&preHashHash){
            DE.Report(DOUBLESHARP)<<location.printToString(sm)<<name;
            preHashHash = false;
            return;
        }
        else {
            preHashHash = false;
        }
    }
```

### 1.5 检查是否重复定义宏

这里只检查是否在同一文件里面重复定义函数。前面已经存储了单个文件里面定义的宏，所以在遇到宏定义时直接检查即可。

```c++
    auto name = MacroNameTok.getIdentifierInfo()->getName();
    auto data = name.data();
    auto ptr = findMacro(name);
    auto location = MD->getLocation();
    if(ptr != macros.end()){
        DiagnosticsEngine &DE = CI.getDiagnostics();
        DE.Report(MULDEF)<<location.printToString(CI.getSourceManager())<<ptr->name;
    }
```

### 1.6 检查宏是否可以用`const`、引用、内联函数来代替

如果`Token`的类型全都为`literal`或者操作符(`plus`等)，则推荐使用`const`；如果`Token`数量只有一个且为`identifier`，则推荐使用引用；对于其它情况，只要不是定义`foreach`宏，或者使用了编译器的宏，都推荐`inline`函数。

```c++
bool shouldBeConstVar(const ArrayRef<Token> tokens){
    if(tokens.size()==0) return false;
    auto ref_begin = tokens.begin();
    auto ref_end = tokens.end();
    while(ref_begin != ref_end){
        if(idName.equals(ref_begin->getName())) return false;
        if(!(ref_begin->isLiteral()||
            isOp(ref_begin->getName())))
            return false;
        ref_begin++;
    }
    return true;

}
bool shouldBeRef(const ArrayRef<Token> tokens){
    if(tokens.size()==0) return false;
    if(tokens.size()==1 && idName.equals(tokens[0].getName()))
        return true;
    return false;
}
bool shouldBeInline(const ArrayRef<Token> tokens){
    ///foreach macro
    int size = tokens.size();
    if(size>5&&strcmp(tokens[0].getName(),"for")
        &&strcmp(tokens[1].getName(),"l_paren")
        &&strcmp(tokens[size-1].getName(),"r_paren"))
        return false;
    ///use compiler macro
    auto ref_begin = tokens.begin();
    auto ref_end = tokens.end();
    while(ref_begin != ref_end){
        if(idName.equals(ref_begin->getName())){
            auto info = ref_begin->getIdentifierInfo();
            if(isCompilerDefine(info->getName().data()))
                return false;
        }
        ref_begin++;
    }
    return true;
}
```





