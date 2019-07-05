lexer grammar C1Lexer;

tokens {
    Comma,
    SemiColon,
    Assign,
    LeftBracket,
    RightBracket,
    LeftBrace,
    RightBrace,
    LeftParen,
    RightParen,
    If,
    Else,
    While,
    Const,
    Equal,
    NonEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,

    Int,
    Float,
    Void,

    Identifier,
    IntConst,
    FloatConst
}

Comma: ',' ;
SemiColon: ';' ;
Assign: '=' ;
LeftBracket: '[' ;
RightBracket: ']' ;
LeftBrace: '{' ;
RightBrace: '}' ;

LeftParen: '(' ;
RightParen: ')' ;

If: 'if' ;
Else: 'else' ; 
While: 'while' ;
Const: 'const' ;
Equal: '==' ;
NonEqual: '!=' ;
Less: '<' ;
Greater: '>' ;
LessEqual: '<=' ;
GreaterEqual: '>=' ;

Plus: '+'  ;
Minus: '-' ;
Multiply: '*' ;
Divide: '/' ;
Modulo: '%' ;

Int: 'int' ;
Float: 'float' ;
Void: 'void' ;

Identifier: [a-zA-Z_] [a-zA-Z0-9_]* ;

IntConst: ([1-9][0-9]* | '0') 
            | (('0x'|'0X') [0-9a-fA-F]+) 
            | ('0' [0-7]+);
FloatConst:  ([0-9]* '.'  [0-9]+ | [0-9]+ '.' ) ([eE] [+-]? [0-9]+)? 
            | [0-9]+ [eE] [+-]? [0-9]+;


LineComment: ('//' | '/' ( '\\' '\r'? '\n')+ '/' ) ~[\n]*? ( '\\' '\r'? '\n' ~[\n]*?)* '\r'? '\n' -> skip ;
BlockCommset: '/*' .*? '*/' -> skip ;


WhiteSpace: [ \t\r\n]+ -> skip ;
