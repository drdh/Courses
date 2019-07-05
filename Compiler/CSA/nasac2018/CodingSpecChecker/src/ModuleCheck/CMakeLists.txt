# add_library(module-check STATIC ModuleCheck.cpp)

# set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/ModuleCheck.exports)
# add_llvm_loadable_module(ModuleCheckPlugin ModuleCheck.cpp PLUGIN_TOOL clang)

# if(LLVM_ENABLE_PLUGINS AND (WIN32 OR CYGWIN))
#   target_link_libraries(ModuleCheckPlugin PRIVATE 
#   	clangAnalysis clangAST clangStaticAnalyzerCore LLVMSupport)
# endif()

add_library(module-check STATIC ModuleCheck.cpp)