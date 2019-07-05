# gen2调研

关于类方法和实例方法,gen2.py使用FuncInfo.is_static标志指示是否是类方法

调用链为:

```mermaid
graph TD
E[startwith static and context = 'class'/'struct'</br>in parse_func_decl</br>line 450 in hdr_parse.py]-->D[static_method</br>in parse_func_decl</br>line 603 in hdr_parse.py]
D[static_method</br>in parse_func_decl</br>line 603 in hdr_parse.py]-->A[\S in decl NO.2</br>in add_func</br>line 950 in gen2]
A[\S in decl NO.2</br>in add_func</br>line 950 in gen2]-->B[is_static=True]
B[is_static=True]-->C[FuncInfo.is_static=True]
```

# 完成python生成c hook代码

继续上周的工作,进行了以下完善

1. 加入二段正则表达式以修正对于部分符号提取返回值的错误
2. 加入balance_split函数用于处理正则式无能为力的括号嵌套匹配部分
3. 将整体封装成一个类Hook

目前经过本周两个晚上对于正则式的调试,绝大部分函数(>99%)都能正确地从符号生成hook代码,接下来的检查和调试需要配合其他部分进行大规模的测试,因此这部分工作暂时完成.