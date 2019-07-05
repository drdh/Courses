# Periscope of C++ & Python
## Members
| Name   | identity | ID         |
| ------ | -------- | ---------- |
| 胡明哲 | 组长     | ----       |
| 陈俊羽 | 组员     | PB15121729 |
| 吴紫薇 | 组员     | PB16110763 |
| 彭昀   | 组员     | PB16001632 |
| 雷婷   | 组员     | PB16051553 |

## Introduction
This project focuses on performance and security analysis and improvement in Python and C/C++ multi-language blending. Today, more and more software and tools provide Python application programming interfaces for popular programming, but Python programs take a mechanism to explain execution, and their performance is not optimistic. To this end, many framework and tool providers use C/C++ for the background implementation, and then use the C/C++ extension mechanism provided by Python to wrap these C/C++ implementation into a Python library for the user to call. For multi-language hybrids, there is potential for improvement in their performance, memory usage, etc. For this purpose, appropriate tools are needed to support cross-language performance, memory usage and so on...

## Task Assignment

#### Chen

- Investigated the use of nm to scan the symbol table of the dynamic link library and automatically generated a c program for hook
- Investigated LD_PRELOAD technology for hook of dynamic link library
- Added graphviz to the project to draw the call graph and output the corresponding information
- Accomplished cross-language data transfer with shared memory set

#### Peng

- Investigated the system memory type and the existing methods to get the memory usage
- Modified the open source tool Memory Profiler and added analysis function for function calls
- Constructed a data structure for cross-language function call

#### Lei

- Investigated the dynamic code injection on Linux
- Investigated the principle and framework of opencv binding

#### Wu

- Investigated OpenCV built-in tracing framework and existing analysis tools
- Summarized the current python-c/c++ issues and possible solutions in different scenarios
- Conducted a survey of the application field of python, commonly used packages and concerns of primary python users
- Built a suitable virtual environment for test

## Discussion

* 10.23

  Project opening and basic introduction

* 11.01

  Project's principle introduction and task assignment

* 11.20

  Discuss data structure design of function call graph and C++ function classification method

* 12.04

  Discuss data structure modification for function call graphs and mangle conversion rules 

* 12.15

  Discussed the principle of gen2.py on C language side wrapper function generation

* 12.23

  Discussed the synthesis of the various parts of the project and the overall framework of the project

 * 01.11

   Writed a technical report and prepared a defense PPT

## Progress

11.20 - 12.04

- Investigated the principle of Memory Profiler and designed function call graph data structure

12.04 - 12.11

- Automated reading of symbol tables and mapping of mangle names

12.11 - 12.20

- Automatically generated function prototypes from the symbol table information, ie hook code

12.20 - 12.27

- Added shared memory support
- Added drawing function using graphviz

12.27 - 1.1

- Refactored the code to make it more in line with software engineering principles

1.2 - 1.11

- Optimized the project, wrote a technical report and prepared a reply PowerPoint

## Problem and Solution

1. **Problem** : C program side and python program side cannot directly transfer information by calling

   **Solution** : use the shared memory mechanism to let c programs and python programs read and write information from the same block of memory

2. **Problem**: Function names in symbol tables is not readable due to c's mangle mechanism, so demangle name is not unique.

   **Solution**: Use the -C parameter of nm to export demangle name as a control, mangle name as the map key value, and demangle name as the user-friendly output data.

3. **Problem**: We cannot get class and module when python function is running

   **Solution**: Indirectly get the class and module in which the function is located by dynamically getting the environment in which the function argument is located

## TODO List

- [ ] Solve the problem that C-style functions can't be properly hooked
- [ ] More friendly call graph design
- [ ] Provide a variety of parameter choices for user runtime
- [ ] Join the memory analysis mechanism
- [ ] Improve tool stability and robustness
- [ ] Expand the testing scope of the tool
- [ ] Add parallel program analysis support

## Link

[Memory Profiler](https://github.com/pythonprofilers/memory_profiler)

[Graphviz](https://github.com/ellson/MOTHBALLED-graphviz)

[Opencv](https://github.com/opencv/opencv)

[Opencv-python](https://github.com/skvark/opencv-python)