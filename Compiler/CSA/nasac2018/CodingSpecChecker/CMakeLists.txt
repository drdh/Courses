cmake_minimum_required(VERSION 3.4.3)

project(NASAC2018Competition VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD            17)
set(CMAKE_CXX_STANDARD_REQUIRED   ON)
set(CMAKE_CXX_EXTENSIONS          OFF)

find_program(LLVM_CONFIG "llvm-config")
if(NOT LLVM_CONFIG)
	message(FATAL_ERROR "llvm-config not found -- ${LLVM_CONFIG}")
endif(NOT LLVM_CONFIG)
message(STATUS "Found LLVM_CONFIG as ${LLVM_CONFIG}")

set(CONFIG_COMMAND ${LLVM_CONFIG} "--includedir" "--libdir" "--libnames" "x86")
execute_process(COMMAND ${CONFIG_COMMAND}
	RESULT_VARIABLE HAD_ERROR OUTPUT_VARIABLE CONFIG_OUTPUT)
if(HAD_ERROR)
	message(FATAL_ERROR "${CONFIG_COMMAND} failed with status ${HAD_ERROR}")	
endif(HAD_ERROR)
string(REGEX REPLACE "[ \t]*[\r\n]+[ \t]*" ";" CONFIG_OUTPUT ${CONFIG_OUTPUT})

list(GET CONFIG_OUTPUT 0 INCLUDE_DIR)
list(GET CONFIG_OUTPUT 1 LIBRARY_DIR)
list(GET CONFIG_OUTPUT 2 LLVM_TARGETS_TO_BUILD)
string(REGEX REPLACE " " ";" LLVM_TARGETS_TO_BUILD ${LLVM_TARGETS_TO_BUILD})

set(LLVM_INCLUDE_DIR ${INCLUDE_DIR} CACHE PATH "Path to llvm/include")
set(LLVM_LIBRARY_DIR ${LIBRARY_DIR} CACHE PATH "Path to llvm/lib")

include_directories("${LLVM_INCLUDE_DIR}")
link_directories("${LLVM_LIBRARY_DIR}")

add_compile_options(-fno-rtti)

add_subdirectory(src)
