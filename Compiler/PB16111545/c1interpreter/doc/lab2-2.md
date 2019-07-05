## 代码的构建

构建过程实在是太麻烦了，顺序也不是完全按照`实验提示`提供的，所以这里仅给出我对一些问题的解决思路。按照函数出现的顺序。

除了通过阅读本实验相关的代码来找到相应的函数，下面的链接对代码构建也非常重要，

[LLVM IR LangRef](http://llvm.org/docs/LangRef.html)

[IRBuilder.h](https://github.com/llvm-mirror/llvm/blob/release_60/include/llvm/IR/IRBuilder.h)

除此之外，使用`vscode`的`C++ Intellisense`也有很大帮助。

最后，通过阅读`ref`产生的IR也可能提示用什么指令，然后可以找到相关的`Createxxx`

### (1)`assembly`

最核心的部分就是遍历`node.global_defs`

```cpp
for(auto &def:node.global_defs){
        def->accept(*this);
}
```

使用访问者模式，在`syntax_tree.h`中可以找到`global_defs`的内容定义，以下内容几乎都会用到这个头文件的信息，不再重复。此处的`accept`也可以在该文件中找到，`virtual void accept(syntax_tree_visitor &visitor) override final;` 

### (2)`func_def`

1. 进入函数体的定义，那么需要设置`in_global=false`

2. 然后检查是否有重复定义。按照文档提示，是使用`std::unordered_map<std::string, llvm::Function *> functions;`结构，查询该结构的方法函数，可以使用`count`

3. 没有重复定义的时候，可以参考`runtime.cpp`里面的函数定义方法

   ```cpp
   auto inputInt_impl = Function::Create(FunctionType::get(Type::getVoidTy(module->getContext()),
                                                            {Type::getInt32PtrTy(module->getContext())},
                                                            false),
                                          GlobalValue::LinkageTypes::ExternalLinkage,
                                          "inputInt_impl",
                                          module);
   ```

   由于没有返回值和参数，所以具体设置为

   ```cpp
   current_function = Function::Create(FunctionType::get(Type::getVoidTy(context), {}, false),
                                       GlobalValue::LinkageTypes::ExternalLinkage,
                                       name, module.get());
   ```

4. 为了之后的函数定义，需要把该定义的函数写入`functions`结构体

5. 进入函数体，先设置一个`BasicBloack`这个在`lab2-1`中有涉及

   ```cpp
   builder.SetInsertPoint(BasicBlock::Create(context,"entry",current_function));
   ```

6. 然后使用访问者模式对`body`进行visit

7. 最后就是设置返回的语句，以及将`in_global`设为`true`

### (3)`cond`

1. 应该是先写的`binop_expr`，而`cond`与之非常相似。

2. 先用`accept`处理`lhs/rhs`注意值的记录以及值的类型记录

3. 与下面很多函数的情况一样，需要分成`int`与`float`情况分别处理

   很容易找到相关的比较函数是

   ```cpp
   value_result = builder.CreateICmpSGT(lhs, rhs)
   value_result = builder.CreateFCmpOGT(lhs, rhs);
   ```

4. 有几个语义检查需要进行

   `float`不能使用`!=,==` 

### (4)`binop_expr`

1. 需要按照`constexpr_expected`分成两部分

   `true`部分可以直接使用本地运算符`+-*/%`直接计算

   `false`部分需要`Createxxx`一些指令

2. 每个部分内部同样需要按照`int`　`float`来区分

3. 需要调用`accept`来获取左右的值与类型

4. 很容易得到相应的指令是

   ```cpp
   value_result = builder.CreateAdd(lhs, rhs);
   value_result = builder.CreateFAdd(lhs, rhs);
   ```

5. 需要做的语义检查是，`float`类型不能进行`%`运算

### (5)`unaryop_expr`

1. 写完`binop_expr`再写这个应该不难。同样先分为`constexpr_expected`然后内部小的方面分为`is_int`

2. 此处只有一个`rhs`的`accept`

3. `constexpr_expected`直接运算就可以了

4. 其他部分很容易找到相关指令为

   ```cpp
   value_result=builder.CreateNeg(value_result);
   value_result=builder.CreateFNeg(value_result);
   ```

### (6)`lval`

1. 这部分构建比较复杂。先取值

   ```cpp
   //变量地址在 LLVM 数据结构中的表示、是否为常量、是否为数组、是否为整型
       auto variable_tuple=lookup_variable(node.name);
       auto lval=std::get<0>(variable_tuple);
       bool is_const=std::get<1>(variable_tuple);
       bool is_array=std::get<2>(variable_tuple);
       bool is_int=std::get<3>(variable_tuple);
   ```

2. 然后进行相关的语义检查，判断是否声明，判断不能为`constexpr_expected`

3. 下面最主要的部分需要按照是否为数组`is_array`分成两个大的部分

4. 对于不是数组的情况　

   1. 先语义检查，在`syntax node`内部的`node.array_index`为`nullptr`才对

   2. 然后按照左值与右值来区分

      1. 右值情况。按照是否为`int`来区分，需要取出值，然后存入`value_result`

         ```cpp
         value_result=builder.CreateLoad(Type::getInt32Ty(context),lval);
         value_result=builder.CreateLoad(Type::getDoubleTy(context),lval);
         ```

      2. 左值情况。先语义检查，左值不能为`const`然后取出地址存入即可

         ```cpp
          value_result=lval;
         ```

5. 对于是数组的情况，大体上的分类与上面一致，需要另外注意的地方如下

   1. 由于有索引，所以需要`accept`这个，然后判断是否是`int`不是就需要报错

   2. 需要取出相应的element, 按照文档提示，指令为`getelementptr`然后仔细查找，可以找到这个函数

      ```cpp
      auto element=builder.CreateGEP(lval,index);//即getelementptr
      ```

      然后`index`可以是`std::vector<Value *>`类型。里面只能有两个值，第一个为`0`第二个为相应的index值，但是此处的值需要是`Value *`类型

   3. 后面的情况与非数组类似

### (7)`literal`

1. 这部分应该是最先构建出来的，相对后面的是最基础的，需要做的如下

2. 按照`constexpr_expected`分成两个部分，每个部分再按照是否为`int`来分

   1. 对于`constexpr_expected`部分，存入`int_const_result`或者`float_const_result`. 注意上面以及下面所有的需要设置`xxx_result`部分的代码，都需要考虑`is_result_int`该如何设置

   2. 对于非的部分，需要存入`value_result`. 具体就是

      ```cpp
      value_result=ConstantInt::get(Type::getInt32Ty(context), node.intConst);
      value_result=ConstantFP::get(Type::getDoubleTy(context), node.floatConst);
      ```

### (8)`var_def`

1. 这部分代码行数非常大，而且之后的测试也对这个部分修改最多。下面简要说说如何构建。

2. 变量的定义需要分为是否`in_global`不同的部分使用的构建方法很不一样。其内部又需要分为是否为`is_array`　然后再其中又需要分为是否为`int`　在最里面还需要分为是否为有初始值，来区分如何初始化

3. `in_global`

   按照文档的说明是这样的

   > 对于全局变量，你应当使用 `global` 指示。它能够在全局区域内声明符号，并在链接时分配数据段空间，具体的使用方式见参考[文档](http://llvm.org/docs/LangRef.html#global-variables)。这里给出一个提示：`LLVM IR`的库接口中与之等价的内容在 [`llvm::GlobalVariable`](https://github.com/llvm-mirror/llvm/tree/release_60/include/llvm/IR/GlobalVariable.h#L42) 类中，你需要使用类似于 `new GlobalVariable` 的方式在当前 `module` 中创建一个全局变量定义。[`GlobalVariable`](https://github.com/llvm-mirror/llvm/tree/release_60/include/llvm/IR/GlobalVariable.h#L42) 所代表的值本身也是一个指向变量的指针，因此和 `alloca` 指令的结果是一致的，可以统一在`lval_syntax`中处理。
   >
   > 为了简单起见，对于全局变量，无论变量为常量或可变量，其初始化表达式均**要求**是常量表达式。因为有此要求，`C1` 的编译中不需要插入比 main 函数早的代码来完成全局变量的初始化，只需要使用构造函数 [`GlobalVariable::GlobalVariable`](https://github.com/llvm-mirror/llvm/tree/release_60/include/llvm/IR/GlobalVariable.h#L61) 中的 `Initializer` 参数指定初始化的常量值即可；根据变量类型不同，它应为一个 `ConstantInt` 、`ConstantFP` 或`ConstantArray`。

   1. 对于不是array的情况，如果没有显式值，初始化为0，如果有，就相应处理，如果类型不同意，就直接进行转化，因为此处不能用变量来初始化全局变量。创建全局变量的方法在`runtime.cpp`中有涉及，可以就直接照搬为

      ```cpp
      var_ptr=new GlobalVariable(*module,Type::getInt32Ty(context),is_const,
                                         GlobalValue::ExternalLinkage,init_value,"");
      var_ptr=new GlobalVariable(*module,Type::getDoubleTy(context),is_const,
                                         GlobalValue::ExternalLinkage,init_value,"");
      ```

   2. 对于是`array`的情况需要额外处理`index`情况。

      需要对`index`做语义检查，不能为`float`，index显式值的大小不能小于初始化列表的长度，index值必须大于零

      初始化列表的值如果类型不符，需要直接做类型转换`int_const_result=float_const_result;`

      然后最核心的地方就是

      ```cpp
      var_ptr=new GlobalVariable(*module.get(),ArrayType::get(Type::getInt32Ty(context),array_length),is_const,                        GlobalValue::ExternalLinkage,ConstantArray::get(ArrayType::get(Type::getInt32Ty(context),array_length),init_array),"");                                           
      var_ptr=new GlobalVariable(*module.get(),ArrayType::get(Type::getDoubleTy(context),array_length),is_const,                                      GlobalValue::ExternalLinkage,ConstantArray::get(ArrayType::get(Type::getInt32Ty(context),array_length),init_array),"");
      ```

4. 非`in_global`

   按照文档的说明是这样的

   > 对于局部变量，你应当通过调用 [IRBuilder::CreateAlloca](https://github.com/llvm-mirror/llvm/tree/release_60/include/llvm/IR/IRBuilder.h#L1156) 创建 `alloca` 指令。这一指令能够在栈上分配空间，具体的使用方式请参考[文档](http://llvm.org/docs/LangRef.html#alloca-instruction)。你需要思考如何使用 `alloca` 指令创建变量并获取指向它的指针值。
   >
   > 对于局部变量，无论变量为常量或可变量，其初始化表达式均**不要求**是常量表达式。

   局部变量的分类与全局变量相同，但是其处理的方式有几个不同之处

   1. 不符的初始化类型需要使用指令来转换

      ```cpp
      value_result=builder.CreateFPToSI(value_result,Type::getInt32Ty(context));
      value_result=builder.CreateSIToFP(value_result,Type::getDoubleTy(context));
      ```

   2. 构建一个变量，就是在获取地址空间

      ```cpp
      var_ptr=builder.CreateAlloca(Type::getInt32Ty(context),nullptr,"");
      var_ptr=builder.CreateAlloca(Type::getDoubleTy(context),nullptr,"");
      ```

   3. 初始化值是相当于`store`指令，需要另外构建

      ```cpp
      builder.CreateStore(value_result,var_ptr);
      ```

   4. 对于数组，上面的获取地址空间变为

      ```cpp
      var_ptr=builder.CreateAlloca(ArrayType::get(Type::getInt32Ty(context),array_length),nullptr,"");
      var_ptr=builder.CreateAlloca(ArrayType::get(Type::getDoubleTy(context),array_length),nullptr,"");
      ```

      初始化每个地址空间需要先取出`element=builder.CreateGEP(var_ptr,index);`然后才能`store`

      这几个函数需要详细查看`llvm`的`IRBuilder.h`才能找到，以及找到相应的参数构建办法

5. 最后的最后，需要把变量放入`std::deque<std::unordered_map<std::string, std::tuple<llvm::Value *, bool, bool, bool>>> variables;`结构，具体的办法可以在文档以及`assembly_builder.h`中找到为

   ```cpp
   bool declare_variable(std::string name, llvm::Value *var_ptr, bool is_const, bool is_array, bool is_int)
   ```

### (9)`assign_stmt`

1. 同样先取值

   ```cpp
   //变量地址在 LLVM 数据结构中的表示、是否为常量、是否为数组、是否为整型
       auto variable_tuple=lookup_variable(node.target->name);
       auto lval=std::get<0>(variable_tuple);
       bool is_const=std::get<1>(variable_tuple);
       bool is_array=std::get<2>(variable_tuple);
       bool is_int=std::get<3>(variable_tuple);
   ```

2. 然后做几个语义检查，判断该变量已经定义，判断`target`不是`const`的，判断关于数组的类型是相符的。

3. 然后`accept` `target`和`value`部分

4. 最后`store`就相当于`assign`　需要注意的是，类型不符的时候需要显式进行类型转换

   ```cpp
   auto cast_value=builder.CreateFPToSI(value,Type::getInt32Ty(context));
   auto cast_value=builder.CreateSIToFP(value,Type::getDoubleTy(context));
   ```

### (10)`func_call`

1. 由于没有参数，所以这部分相对容易

2. 首先语义检查是否定义了函数。这部分在函数的定义里面已经做了。

3. 然后构建`func call`指令就可以了

   ```cpp
   builder.CreateCall(functions[name],{});
   ```

### (11)`block`

1. 这部分最难注意到的应该是环境`scope`的进出。在`assembly_builder.h`中有相关的函数

   ```cpp
   void enter_scope() { variables.emplace_front(); }
   void exit_scope() { variables.pop_front(); }
   ```

2. 然后遍历`body`即可，相应内容可在`syntax_tree.h`中找到

   ```cpp
   for(auto &body:node.body){
           body->accept(*this);
       }
   ```

### (12)`if_stmt`

1. 这部分和`while`也比较麻烦，需要结合`lab2-1`的`BasicBlock`部分以及`SetInsertPoint`来理解与构建。

2. 最完整的是分成四个`bb`, `cond`用于比较判断，`then`用于条件为`true`的情况，`else`用于条件为`false`情况，`next`表示执行完了某一条语句后跳转的部分。

3. 有的`if`语句没有`else`部分，删掉相应的块即可。所以需要分成两个部分。

4. 最完整的包括`else`的构建应该是

   ```cpp
   builder.CreateBr(cond_block);
   //cond
   builder.SetInsertPoint(cond_block);
   node.pred->accept(*this);
   builder.CreateCondBr(value_result,then_block,else_block);
   //then
   builder.SetInsertPoint(then_block);
   node.then_body->accept(*this);
   builder.CreateBr(next_block);
   //else
   builder.SetInsertPoint(else_block);
   node.else_body->accept(*this);
   builder.CreateBr(next_block);
   //next
   builder.SetInsertPoint(next_block);
   ```

5. 需要注意的是，在执行完`then`之后，需要直接跳转到`if`语句之后的部分。

### (13)`while_stmt`

1. 这部分也比较麻烦。

   ```cpp
   builder.CreateBr(cond_block);
   //while_cond
   builder.SetInsertPoint(cond_block);
   node.pred->accept(*this);
   builder.CreateCondBr(value_result,true_block,next_block);
   //while_true
   builder.SetInsertPoint(true_block);
   node.body->accept(*this);
   builder.CreateBr(cond_block);
   //while_next
   builder.SetInsertPoint(next_block);
   ```

2. 需要注意的是，在执行完内部语句之后，需要设置一个跳转直接到判断的地方，这才是循环。

### (14)`empty_stmt`

1. 空即可

## 细节记录

此处记录常用的几条语句，便于代码构建

### 错误处理

```cpp
err.error(node.line,node.pos,"multiple declaration: "+name);
error_flag=true;
```

### 获得Type

```cpp
Type::getInt32Ty(context)
Type::getDoubleTy(context)
```

### 获得常量

```cpp
ConstantInt::get(Type::getInt32Ty(context), node.intConst)
ConstantFP::get(Type::getDoubleTy(context), node.floatConst)
```

### 类型转换

```cpp
auto cast_value=builder.CreateFPToSI(value,Type::getInt32Ty(context));
auto cast_value=builder.CreateSIToFP(value,Type::getDoubleTy(context));
```





## 实验代码的编译和运行

为了不污染环境，单独在工作目录安装

在`/c1recognizer/build`目录下

```bash
cmake -DCMAKE_INSTALL_PREFIX=/home/drdh/lx/Compiler/c1recognizer_install ..
make install
```

在`c1interpreter`下

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DLLVM_DIR=/home/drdh/lx/Compiler/LLVM/llvm-install/lib/cmake/llvm -Dc1recognizer_DIR=/home/drdh/lx/Compiler/c1recognizer_install/cmake ..
```

同时在仓库的根目录

```bash
echo "c1interpreter/build/*" >> .gitignore
```

然后编译

```bash
make -j
```

运行方式

```bash
./c1i -emit-llvm test.c1
./c1i test.c1
```

## 测试与修正

为了便于测试，构造脚本如下

```bash
#!/bin/bash
rm -rf user.ll ref.ll
./c1i -emit-llvm ../test/$1.c1 >user.ll
./c1i ../test/$1.c1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./Libs_for_c1i_ref 
./c1i_ref_ubuntu -emit-llvm ../test/$1.c1> ref.ll
./c1i_ref_ubuntu ../test/$1.c1
diff user.ll ref.ll
```

### 1.

文档中给出的测试代码有误

```cpp
void main()
{
  output_var = 10;
  output();
}
```

由于运行时的函数为

```cpp
int input_ivar;
float input_fvar;
void inputInt();
void inputFloat();
int output_ivar;
float output_fvar;
void outputInt();
void outputFloat();
```

所以应该改为

```cpp
void main()
{
  output_ivar = 10;
  outputInt();
}
```

同时给出的参考可执行文件也有一个小错：将`inputFloat()`误定义为`inpuFloat()`但是不影响运行

### 2.

首先是`var_def`

#### 2.1 

使用最简单的测试

```cpp
void main(){
    int a;
    float b;
}
```

使用`dif`f发现与参考文件的几个不同点

1. 局部变量没有初始化
2. 变量的命名。我是使用`%a`而参考为`%0`
3. `entry`的使用不同

第一点增加初始值设定即可

```cpp
builder.CreateStore(ConstantInt::get(Type::getInt32Ty(context),0),var_ptr);
builder.CreateStore(ConstantFP::get(Type::getDoubleTy(context),0),var_ptr);
```

第二点是因为我的变量声明是这样的

```cpp
var_ptr=builder.CreateAlloca(Type::getDoubleTy(context),nullptr,var_name);
```

为了保持一致性，将其改为

```cpp
var_ptr=builder.CreateAlloca(Type::getDoubleTy(context),nullptr,"");
```

第三点同样是为了保持一致性

从（这是文档上的标准推荐）

```cpp
builder.SetInsertPoint(BasicBlock::Create(context,"entry_BB"+std::to_string(bb_count++),current_function));
```

改为

```cpp
builder.SetInsertPoint(BasicBlock::Create(context,"entry",current_function));
bb_count++;
```

然后完全一致。

#### 2.2

测试改为

```cpp
void main(){
    int a;
    float b;
    a=1;
    b=1;
    int c=a+b;
}
```

发现不同

```
52,53c52,54
<   %5 = add nsw i32 %3, double %4
<   store i32 %5, i32* %2
---
>   %5 = sitofp i32 %3 to double
>   %6 = fadd double %5, %4
>   store double %6, i32* %2
```

参考是先将`int`转化为`float`, 然后使用`fadd`相加，然后`store`的方式为`double`

我一开始处理的时候偷懒了。修改`binop`部分。

修改的地方有三个

1. 在`lval`开头加上`is_result_int=is_int`

2. 在`binop_expr`中`non const`而且至少一个为`float point`后加上

   ```cpp
   if(is_int_left){//先转化
   	lhs=builder.CreateSIToFP(lhs,Type::getDoubleTy(context));
   }
   if(is_int_right){
   	rhs=builder.CreateSIToFP(rhs,Type::getDoubleTy(context));
   }
   ```

3. 取消`var_def`中`local`的初始值与变量类型不对应报错。ref已经改成了类型变化的情况。

   所以需要增加类型转换。

   ```cpp
   value_result=builder.CreateFPToSI(value_result,Type::getInt32Ty(context));
   ```

   其他情况类似(local情况)。

   对于global情况，就直接强制转换就行

   ```cpp
   int_const_result=float_const_result;
   ```

   

#### 2.3 

测试

```cpp
int k[12];
```

我的输出为

```cpp
%19 = getelementptr [12 x i32], [12 x i32]* %18, i32 0, i32 0
store i32 0, i32* %19
```

参考为

```cpp
%19 = getelementptr [12 x i32], [12 x i32]* %18, i32 0
store i32 0, [12 x i32]* %19
```

根据其文档，猜想原因是，我使用的是`std::vector<Value*>index`而实际上可以直接使用`const`来声明数组

于是从

```cpp
element=builder.CreateGEP(var_ptr,index);
```

改为

```cpp
element=builder.CreateConstGEP1_32(var_ptr,i);
```

其中`int`与`float`同样处理**后来经反馈，发现是ref自己的错误，并且导致下面的2.4错误，复原之后无事**

然后将`index`变量部分注释掉。

#### 2.4

测试

```cpp
int k[7]={1,2,3}；
g=k[3];
```

发现输出的`ll`内容一样，但是在使用` ./c1i ../test/$1.c1`执行自己的代码的时候出现`Segmentation fault`猜测可能是运行时候的问题。

另外，使用如下代码测试，ref也出现`Segmentation fault`

```cpp
void main(){
    int k[8]={1,2,3};
    k[3]=8;
}
```

尝试了很久都不知道怎么解决。后来发现**是2.3的错误修改，已改正**

#### 2.5

```cpp
void main(){
    float a=2;
    float b=-a;
}
```

出现的不同是

```
<   %3 = fsub double -0.000000e+00, %2
---
>   %3 = fsub double 0.000000e+00, %2
```

未知

#### 2.6

测试如下

```cpp
float a=12;
void main(){   
    output_fvar=a;
    outputFloat();   
}
```

我的报错是

```
Error at position 3:4 Constexper cannot be Lval
Error at position 3:16 Constexper cannot be Lval
Semantic failed. Exiting.
Error at position 3:4 Constexper cannot be Lval
Error at position 3:16 Constexper cannot be Lval
Semantic failed. Exiting.
```

仔细调查后发现的原因是，在使用完`constexpr=true`之后，并没有对其还原。



### 3.

#### 3.1

测试`if/else` `while`等等

最简单的测试

```cpp
void main(){
    int a=0;
    if(a>0){
        int b=9;
    }
}
```

发现有一些标签上的差别

```
23c23
< define void @inputFloat() {
---
> define void @inpuFloat() {
45c45
<   br label %cond_BB1
---
>   br label %IfB
47c47
< cond_BB1:                                         ; preds = %entry
---
> IfB:                                              ; preds = %entry
50c50
<   br i1 %2, label %then_BB2, label %else_BB3
---
>   br i1 %2, label %ThenB, label %AfterIf
52c52
< then_BB2:                                         ; preds = %cond_BB1
---
> ThenB:                                            ; preds = %IfB
55c55
<   br label %next_BB4
---
>   br label %AfterIf
57,60c57
< else_BB3:                                         ; preds = %cond_BB1
<   br label %next_BB4
< 
< next_BB4:                                         ; preds = %else_BB3, %then_BB2
---
> AfterIf:                                          ; preds = %ThenB, %IfB
```

同样为了一致性，原来为

```cpp
auto cond_block=BasicBlock::Create(context,"cond_BB",current_function);
auto then_block=BasicBlock::Create(context,"then_BB",current_function);
auto else_block=BasicBlock::Create(context,"else_BB",current_function);
auto next_block=BasicBlock::Create(context,"next_BB",current_function);
```

修改为[见源码]，重新按照是否有`else_body`来分割代码

#### 3.2

测试`while`同样发现有不同的标签问题。同样也发现ref对于`bb_count`的计数，仅仅从`while`才有，其他地方都没有。。。很不规范。

同时也发现另外一个问题，对于`a=a-1`

我的是

```
%34 = sub nsw i32 %33, 1
```

ref是

```
%34 = sub i32 %33, 1
```

但是对于`a=b-1`却没有差别。

尝试后，将

```cpp
value_result = builder.CreateNSWAdd(lhs, rhs);
```

改为

```cpp
value_result = builder.CreateAdd(lhs, rhs);
```

可行。查看文档发现

> `nuw` and `nsw` stand for “No Unsigned Wrap” and “No Signed Wrap”, respectively. If the `nuw` and/or `nsw` keywords are present, the result value of the `sub` is a [poison value](http://llvm.org/docs/LangRef.html#poisonvalues) if unsigned and/or signed overflow, respectively, occurs.

没看懂，暂放。

### 4.

#### 4.1

ref 对浮点数的`==`与`!=`与`%`没有报错。ref错误。

#### 4.2

ref 对global情况下的初始化

```cpp
float f[3]={1,2,3};
```

没有做类型转化，而导致没有初始化，ref错误。

## 总结

这项实验花了非常长的时间。需要阅读大量相关的代码，文档等等。同时也认识到，C++的类型系统对于代码构建真的非常重要。用一些小的测试来帮助一点点构建代码，一点点完善代码；格外注意一些全局变量的存在，虽然它们的存在能简化代码，但是非常容易就忘记了设置；差不多完成代码后，需要用大量有目的，有针对性的测试来debug, 测试也是一点一点进行，不能一次性大量相关测试，这样很难找到错误的根源。