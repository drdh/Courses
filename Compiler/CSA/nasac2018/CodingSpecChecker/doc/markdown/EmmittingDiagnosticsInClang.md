## 在Clang中发射诊断信息

诊断信息需要做到：清晰、正确、精准并且可读。Clang在诊断信息处理上做了很多的努力，可以阅读[Expressive Diagnostics](https://clang.llvm.org/diagnostics.html)进一步了解。

### Clang中诊断信息相关的类型定义

#### DiagnosticsEngine

[DiagnosticEngine](https://clang.llvm.org/doxygen/classclang_1_1DiagnosticsEngine.html)是Clang诊断系统中最核心的类，它主要的作用是提供[Report](https://clang.llvm.org/doxygen/classclang_1_1DiagnosticsEngine.html#a03686c59442babd725417ff740b397b4)方法给用户提交诊断信息，并提供设置诊断报告行为属性的设置能力。

**定制诊断信息** 在发射一条定制诊断信息前，用户必须通过方法[getCustomDiagID](https://clang.llvm.org/doxygen/classclang_1_1DiagnosticsEngine.html#a0a8521bd7fd1c68c33f65775b7ee85be)获取一个对应的定制诊断信息ID，提供[诊断信息级别](https://clang.llvm.org/doxygen/classclang_1_1DiagnosticsEngine.html#a94e5078973aa3a34432e91f3b26263e0)和信息格式字符串，获取ID的目的是方便用户重用诊断信息。

**诊断信息格式化** Clang针对诊断信息提供了[格式化语言](https://clang.llvm.org/docs/InternalsManual.html#the-format-string)，提供各种占位符实现格式字符串用以构建最终输出的诊断信息，示例如下：

```c++
const unsigned ID = DE.getCustomDiagID(
    clang::DiagnosticEngine::Warning,
	"Pointer variable '%0' should have a 'p_' prefix");
DE.Report(SourceLocation, ID).AddString(NamedDecl.getName());
```

#### DiagnosticBuilder

[DiagnosticBuilder](https://clang.llvm.org/doxygen/classclang_1_1DiagnosticBuilder.html)是DiagnosticEngine调用Report方法后返回的对象类型，主要负责诊断信息格式字符串的占位符替换。DiagnosticBuilder对象所代表的诊断信息会在析构时自动发射。

#### DiagnosticsConsumer

Clang诊断系统的设计采用了观察者模式，当一个诊断信息被发射后，会通知所有注册的[DiagnosticConsumer](https://clang.llvm.org/doxygen/classclang_1_1DiagnosticConsumer.html)对象进行处理，这些对象可以把诊断信息打印到终端或者记录到日志文件。在基于Clang开发工具时，已经提供的Consumer子类已经能满足绝大部分的需求，很少需要定制。

### 使用注意事项

FrontendAction默认使用TextDiagnosticPrinter作为DiagnosticsConsumer，DiagnosticsConsumer需要实现`BeginSourceFile`和`EndSourceFile`两个回调函数。使用时需要知道FrontendAction调用这两个回调函数的时机，当FrontendAction在调用`EndSourceFile`方法后在调用DiagnosticsEngine的`Report`方法则会触发错误。







