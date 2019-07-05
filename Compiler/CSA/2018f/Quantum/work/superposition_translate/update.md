# Update 19.01.06
作者：李权熹
#### 新增内容
- Makefile
- 类`Instruction`中：
  - `bool input(std::istream &is = std::cin)`输入，部分取代原Lexer的功能
  - `bool isEnd()`判断是否为结尾
  - `bool isOperation()`
  - `bool isEmpty()`
#### 文件变化
- 类`Qubit`、`Instruction`单独列成头文件
- 创建`Level.h`用于声明函数和类，实现放在`levelize.cpp`
- `translate.cpp`作为源文件不在被include，`translate`函数的声明被放在`quantum.h`
- `choose.cpp`改名为`Lexer.h`
- 删掉`code_prepare.h`（这个文件里就一句`#include levelize.cpp`）
#### 接口变化
- 类`Qubit`、`Instruction`中一些字段变为私有
- `choose.cpp`中：
  - `void print_qubit()`变为`void Qubit::print() const`
- `class Qubit`中：
  - `char *getname()`变为`const char *getName() const`
  - 添加`getInstructions`方法
- `class Instruction`中
  - `void setprefix(char *)`变为`void setPrefix(const char*)`
  - `Qubit* get_qubit(int i)`改为`Qubit* getQubit(int i)`
  - `void print()`变为`void print(std::ostream &os = std::cout) const`
  - `void printcode(std::ostream &os = std::cout)`变为`void printCode(std::ostream &os = std::cout) const`
  - `bool judge_qubits()`变为`bool judgeQubits() const`
  - `bool set_qubits()`变为`bool setQubits()`

#### 实现方法变化
- `Instruction::prefix`类型改为`std::string`
- 类`Qubit`中：
  - `name`类型改为`std::string`
  - `Instruction queue[]`改为`std::vector<Instruction*> instructions`
- 删除`cutoperator`和`getqubit`，改用`Intruction::getSubstr`
- `choose.cpp`中`judge`和`hashi`移动到`Instruction.h`中（作为静态方法）
- `choose.cpp`中`Operation`,`Measure`,`Decline`的定义移动到`Instruction.h`中，封装为枚举`Instruction::InstructionType`

#### 其他
- `Qubit`、`Instruction`中成员函数改为内联
- `Qubit`、`Instruction`中成员函数命名统一为小驼峰式
