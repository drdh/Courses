# G3 两开花队 - DefineMagic
- 队员
  - PB16001676-吴永基：[wyj317@mail.ustc.edu.cn](mailto:wyj317@mail.ustc.edu.cn)
  - PB16110428-王浩宇：[tzwhy@mail.ustc.edu.cn](mailto:tzwhy@mail.ustc.edu.cn)
  - PB16120156-邵军阳：[sjygd@mail.ustc.edu.cn](mailto:sjygd@mail.ustc.edu.cn)
  - PB116110766-陆万航：[rg1336@mail.ustc.edu.cn](mailto:rg1336@mail.ustc.edu.cn)
- 目录名：G3-DefineMagic
- 选题： c语言变量定义的优化：包括c语言变量的命名规范及检查、结构体存储空间的极小化（对对齐问题的优化处理）、较长布尔表达式的换行处理。
- 说明
  1. c语言变量命名的规范有助于增加代码的可读性和安全性。主要优化方向有：对作用域内同名的变量进行重新命名和对变量进行分词重构，增强变量的规范命名。
  2. 由于c语言的对齐方式问题，结构体变量的存储空间可能未达到极小化，通过调整结构体内部的变量命名顺序，可以降低结构体变量的存储开销。
  3. 进行条件判断时，由于判断的条件可能会较多，if语句中的布尔表达式可能很多，因此可以对较长的布尔表达式进行合理的换行，使得条件判断的内容能以一个较为清晰的模式呈现。
- 命名检查部分在目录NameChecker中，条件判断赋值检查部分在目录IfStmtAssignChecker中。