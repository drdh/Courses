# 1

> 了解并总结以下三个函数的作用、接口和主要处理流程；
>
> - enterRecursionRule
> - unrollRecursionContexts
> - adaptivePredict

## enterRecursionRule

`externals/antlr4cpp/src/antlr4cpp/runtime/Cpp/runtime/src/Parser.cpp`

```cpp
void Parser::enterRecursionRule(ParserRuleContext *localctx, size_t ruleIndex) {
  enterRecursionRule(localctx, getATN().ruleToStartState[ruleIndex]->stateNumber, ruleIndex, 0);
}

void Parser::enterRecursionRule(ParserRuleContext *localctx, size_t state, size_t /*ruleIndex*/, int precedence) {
  setState(state);
  _precedenceStack.push_back(precedence);
  _ctx = localctx;
  _ctx->start = _input->LT(1);
  if (!_parseListeners.empty()) {
    triggerEnterRuleEvent(); // simulates rule entry for left-recursive rules
  }
}
```

这是一个重载函数。

作用：为递归开始做准备。

接口：

```cpp
void Parser::enterRecursionRule(ParserRuleContext *localctx, size_t state, size_t /*ruleIndex*/, int precedence) 
```

流程：首先它设置了一个 DFA 的状态，然后把调用者的优先级压入优先级栈``_precedenceStack``中。然后设置start，最后，如果parseListener不为空，就触发`triggerEnterRuleEvent`

## unrollRecursionContexts

```cpp
void Parser::unrollRecursionContexts(ParserRuleContext *parentctx) {
  _precedenceStack.pop_back();
  _ctx->stop = _input->LT(-1);
  ParserRuleContext *retctx = _ctx; // save current ctx (return value)

  // unroll so ctx is as it was before call to recursive method
  if (_parseListeners.size() > 0) {
    while (_ctx != parentctx) {
      triggerExitRuleEvent();
      _ctx = dynamic_cast<ParserRuleContext *>(_ctx->parent);
    }
  } else {
    _ctx = parentctx;
  }

  // hook into tree
  retctx->parent = parentctx;

  if (_buildParseTrees && parentctx != nullptr) {
    // add return ctx into invoking rule's tree
    parentctx->addChild(retctx);
  }
}
```

作用：结束递归后的处理。

接口：

```cpp
void Parser::unrollRecursionContexts(ParserRuleContext *parentctx)
```

流程：pop栈_precedenceStack，设置stop内容，保存当前ctx, 当parseListener不为空，触发结束时间。将返回的ctx挂到树上。

## adaptivePredict

`externals/antlr4cpp/src/antlr4cpp/runtime/Cpp/runtime/src/atn/ParserATNSimulator.cpp`

```cpp
size_t ParserATNSimulator::adaptivePredict(TokenStream *input, size_t decision, ParserRuleContext *outerContext) {

#if DEBUG_ATN == 1 || DEBUG_LIST_ATN_DECISIONS == 1
    std::cout << "adaptivePredict decision " << decision << " exec LA(1)==" << getLookaheadName(input) << " line "
      << input->LT(1)->getLine() << ":" << input->LT(1)->getCharPositionInLine() << std::endl;
#endif

  _input = input;
  _startIndex = input->index();
  _outerContext = outerContext;
  dfa::DFA &dfa = decisionToDFA[decision];
  _dfa = &dfa;

  ssize_t m = input->mark();
  size_t index = _startIndex;

  // Now we are certain to have a specific decision's DFA
  // But, do we still need an initial state?
  auto onExit = finally([this, input, index, m] {
    mergeCache.clear(); // wack cache after each prediction
    _dfa = nullptr;
    input->seek(index);
    input->release(m);
  });

  dfa::DFAState *s0;
  if (dfa.isPrecedenceDfa()) {
    // the start state for a precedence DFA depends on the current
    // parser precedence, and is provided by a DFA method.
    s0 = dfa.getPrecedenceStartState(parser->getPrecedence());
  } else {
    // the start state for a "regular" DFA is just s0
    s0 = dfa.s0;
  }

  if (s0 == nullptr) {
    bool fullCtx = false;
    std::unique_ptr<ATNConfigSet> s0_closure = computeStartState(dynamic_cast<ATNState *>(dfa.atnStartState),
                                                                 &ParserRuleContext::EMPTY, fullCtx);

    _stateLock.writeLock();
    if (dfa.isPrecedenceDfa()) {
      /* If this is a precedence DFA, we use applyPrecedenceFilter
       * to convert the computed start state to a precedence start
       * state. We then use DFA.setPrecedenceStartState to set the
       * appropriate start state for the precedence level rather
       * than simply setting DFA.s0.
       */
      dfa.s0->configs = std::move(s0_closure); // not used for prediction but useful to know start configs anyway
      dfa::DFAState *newState = new dfa::DFAState(applyPrecedenceFilter(dfa.s0->configs.get())); /* mem-check: managed by the DFA or deleted below */
      s0 = addDFAState(dfa, newState);
      dfa.setPrecedenceStartState(parser->getPrecedence(), s0, _edgeLock);
      if (s0 != newState) {
        delete newState; // If there was already a state with this config set we don't need the new one.
      }
    } else {
      dfa::DFAState *newState = new dfa::DFAState(std::move(s0_closure)); /* mem-check: managed by the DFA or deleted below */
      s0 = addDFAState(dfa, newState);

      if (dfa.s0 != s0) {
        delete dfa.s0; // Delete existing s0 DFA state, if there's any.
        dfa.s0 = s0;
      }
      if (s0 != newState) {
        delete newState; // If there was already a state with this config set we don't need the new one.
      }
    }
    _stateLock.writeUnlock();
  }

  // We can start with an existing DFA.
  size_t alt = execATN(dfa, s0, input, index, outerContext != nullptr ? outerContext : &ParserRuleContext::EMPTY);

  return alt;
}
```

作用：用DFA预测

接口：

```cpp
size_t ParserATNSimulator::adaptivePredict(TokenStream *input, size_t decision, ParserRuleContext *outerContext)
```

流程：找到DFA, 找初始状态s0, 开始预测。

# 2

> - 描述ANTLR的错误处理机制；

reference: 《antlr4权威指南》

- 默认情况下，ANTLR将所有的错误信息发送至标准错误，可以通过实现接口`ANTLRErrorListener`改变消息的目标输出和内容
- 必要情况下，语法分析器在遇到无法匹配词法符号的错误时，执行单词法符号补全和单词法符号移除。如果这些方案不奏效，词法分析器将向后查找词法符号，直到它遇到一个符合当前规则的后续部分的合理词法符号为止，接着词法分析器将会继续词法分析。
- 一些语法错误十分常见，对他们进行特殊处理，比如在嵌套的函数调用后写错 括号的数量。
- 可以修改默认的错误处理。

# 3

> - 对比分析SLL、LL、以及ALL(*)的异同，例如文法二义、试探回溯等

- SLL所能表示的语法是 LL(k) 表示的语法的子集，每次解析的决定都只基于当前非终结符的下 k 个字符

- SLL 是 ALL(*) 中的一种分析策略，功能不如 LL(\*) 分析器强大。当 SLL 解析失败了，说明有可能是语法错误，也可能是 SLL 的能力有限，这个时候就会切换到全功能模式
- 文法二义：三者都是用产生式的优先级和结合性来处理
- 试探回溯：LL 支持回溯，但是 SLL，ALL(*)去掉了回溯，取而代之的是：在碰到多个可选分支的时候，会为每一个分支运行一个子解析器，每一个子解析器都有自己的DFA，这些子解析器以伪并行的方式探索所有可能的路径，当某一个子解析器完成匹配之后，它走过的路径就会被选定，而其他的子解析器会被杀死。