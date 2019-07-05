# 关于quantum.h中所有可用的API的说明
quantum.h位于src文件夹内，使用时务必将src内的所有文件一起拷贝并且存放于同一目录下。使用时在头文件包含"quantum.h"即可。
## 类
### Qubit类
代表一个qubit。
#### 位置
- `Qubit.h`
- `Qubit.cpp`

#### 方法
##### getName
获得Qubit的名字。
```C++
const std::string &getName() const;
```
##### getInstructions
获取引用该Qubit的所有Instruction
```C++
const std::vector<Instruction *> &getInstructions() const;
```
##### insert
添加引用该Qubit的一条Instruction
```C++
void insert(Instruction* n);
```

#### 公开的字段
##### int mark
没有必要意义可自行使用
##### int sign
这个qubit是否有效。为0则是一个不存在的qubit。由于基本靠数组实现，相当于字符串的'\0'。(希望重构)

### Instruction类
代表一句指令，比如`CX a[1],a[2];`、`qreg a[3]; `等等。
#### 位置
- `Instruction.h`
#### 方法
##### getQubit
获取第i个Qubit
```C++
Qubit* getQubit(int i);
```
##### getQubits
获取引用的所有Qubit，按先后顺序排列
```C++
const std::vector<Qubit *> &getQubits()；
```
##### isOperation
是否为Operation
```C++
bool isOperation();
```
##### print
打印Qubit，默认输出流为`std::cin`
```C++
void print(std::ostream &os = std::cout) const;
```
##### getLine
获取行号
```C++
int getLine();
```
##### input
从输入流（默认`std::cin`）读取一条指令，失败或读到`end`则返回`false`，否则返回`true`
```C++
inline bool input(int &u, int &line, Qubit *qubits, std::istream &is = std::cin);
```
参数含义
- u: 当前已使用qubit数
- line: 行号
- qubits: qubit列表
##### judgeQubit
判断是否所有比特的mark位都不为1，是则返回true
```C++
bool judgeQubits() const;
```
##### setQubit
将所有qubit的信息置为state
```C++
void setQubits(int state);
```
#### 公开的字段
##### int line
指令所在函数
##### int mark
无意义，可作任意用途
##### Instruction* next
下一条指令

### Level类
代表一个层，即一个bundle。定义于levelize.cpp内，内部使用链表串联起该层的所有指令。内部变量和方法
- int line<br>
  对代码经过分层后该层所在的层数
- int length<br>
  这一层指令的数量
- bool is_operation<br>
  这一层是否为门操作
- void insert(Instruction* a)<br>
  往层内插入一个Instruction
- void del(Instruction* a)<br>
  删除层内的一个Instruction。如果不存在于该层内则什么都不做
- Instruction* find_overlap_instruction(set<Qubit*> Q)
  寻找重叠的指令。传入一个Qubit的set集合，返回和该集合有相交的Instruction指针，没有则返回NULL。有多个的情况下返回找到的第一个。
- void print()
  打印该层信息

## 可使用的函数
- void print_qubit(Qubit a)<br>
  打印一个qubit的信息，定义在choose.cpp
- void Lexer(Instruction codes[], Qubit qubits[], int& total_qubit, int& total_instruction)<br>
  文本分解函数。
  定义于choose.cpp。调用方法是传入一个Instruction数组和一个Qubit数组，然后会分解从输入缓冲区读取的代码，并储存到这两个数组里。后两个参数会填入qubit和instruction的数量<br>
  注意：为了和行数对应，codes数组是从1开始，而qubits数组从0开始。<br>
  调用例子：<br>
代码文件execute.cpp
```cpp
#include "quantum.h"
main() {
    Instruction codes[N];
    Qubit qubits[50];
    int qubit_num, instruction_num;
    Lexer(codes, qubits, qubit_num, instruction_num);
    for (int i = 0; i < qubit_num; i++) { 
        print_qubit(qubits[i]);
    }
    for (int i = 1; i <= instruction_num; i++) {  
        codes[i].print();
    }
}
```
新建文本test，在文本中输入
```
qreg q[5];
creg c[3];
h q[0];
h q[1];
h q[2];
s q[0];
s q[1];
s q[2];
cx q[1],q[2];
tdg q[2];
cx q[0],q[2];
t q[2];
```
编译

    g++ execute.cpp -o execute

并且在终端中输入

    ./execute <test

后会得到输出
```
q[0]: 3 6 11
q[1]: 4 7 9
q[2]: 5 8 9 10 11 12
1: qreg q[5];
2: creg c[3];
3: h q[0];
4: h q[1];
5: h q[2];
6: s q[0];
7: s q[1];
8: s q[2];
9: cx q[1],q[2];
10: tdg q[2];
11: cx q[0],q[2];
12: t q[2];
```
- int levelize(Level tower[], Instruction codes[], Qubit qubits[])<br>
  分层化函数，传入经过上一步Lexer函数得到的Instruction，Qubit以及定义的Level tower[],会将代码分层化并装入tower中，同样为了与行数对应，tower数组从1开始。
调用例子：
```cpp
#include "quantum.h"
#define N 300
main() {
    Instruction codes[N];
    Qubit qubits[50];
    int total_qubits,total_ins;
    Lexer(codes, qubits,total_qubits,total_ins);
    Level tower[N];
    int total_level = levelize(tower, codes, qubits);
    printf("分层处理后：\n\n");
    for (int i = 1; i < total_level; i++) tower[i].print();
}
```
对与Lexer中的test进行同样的操作后会得到
```
分层处理后：

1: qreg q[5];
2: creg c[3];
3: h q[0]; h q[1]; h q[2];
4: s q[0]; s q[1]; s q[2];
5: cx q[1],q[2];
6: tdg q[2];
7: cx q[0],q[2];
```

- int translate(Level tower[])<br>
  代码变换函数，传入定义好的Level后会自动对代码进行变换并且存入tower数组中，并返回总的层数。为了与行数对应，下表同样从1开始。有关Instruction和Level的数组均为从1开始。
  调用例子：（该函数内部调用了levelize和Lexer，所以有一些打印）
```cpp
#include "quantum.h"
#define N 300
main() {
    Level tower[N];
    int total_level = translate(tower);
    printf("\n重新打印代码:\n");
    for (int i = 1, line = 1; i <= total_level; i++) {
        auto p = tower[i].head->next;
        while (p) {
            printf("%d: ", line);
            line++;
            p->printcode();
            printf("\n");
            p = p->next;
        }
    }
}
```
同样对test文件进行操作

    .\execute <test

会得到
```
分层处理后：

1: qreg q[5];
2: creg c[3];
3: h q[0]; h q[1]; h q[2];
4: s q[0]; s q[1]; s q[2];
5: cx q[1],q[2];
6: tdg q[2];
7: cx q[0],q[2];

q[0]: 3 4 7

q[1]: 3 4 5

q[2]: 3 4 5 6 7 8


代码变换后后：

1: qreg q[5];
2: creg c[3];
3: h q[1]; h q[2];
4: s q[1]; s q[2];
5: cx q[1],q[2]; h q[0];
6: tdg q[2]; s q[0];
7: cx q[0],q[2];
8: t q[2];

重新打印代码:
1: qreg q[5];
2: creg c[3];
3: h q[1];
4: h q[2];
5: s q[1];
6: s q[2];
7: cx q[1],q[2];
8: h q[0];
9: tdg q[2];
10: s q[0];
11: cx q[0],q[2];
12: t q[2];
```
    