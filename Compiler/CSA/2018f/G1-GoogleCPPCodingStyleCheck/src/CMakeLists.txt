set(CCHECK_RESOURCE_DIR "share/ccheck")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/${CCHECK_RESOURCE_DIR}")
add_subdirectory(ClassCheck)
add_subdirectory(StmtCheck)
add_subdirectory(PPCheck)
add_subdirectory(DeclCheck)

add_executable(google-cpp-coding-style-checker
	ToolDriver.cpp
	# StateStore.cpp
	)
install(TARGETS google-cpp-coding-style-checker DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/")

set(CHECK_ACTION_LIBS
	class-check
	stmt-check
	ppchecker
	decl-check
	)

set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  LLVMOption
  LLVMSupport
  )

set(CLANG_LIBS
	clangAnalysis
	clangARCMigrate
	clangAST
	clangASTMatchers
	clangBasic
	clangCodeGen
	clangCrossTU
	clangDriver
	clangDynamicASTMatchers
	clangEdit
	clangFormat
	clangFrontend
	clangFrontendTool
	clangHandleCXX
	clangHandleLLVM
	clangIndex
	clangLex
	clangParse
	clangRewrite
	clangRewriteFrontend
	clangSema
	clangSerialization
	clangStaticAnalyzerCheckers
	clangStaticAnalyzerCore
	clangStaticAnalyzerFrontend
	clangTooling
	clangToolingASTDiff
	clangToolingCore
	clangToolingInclusions
	clangToolingRefactor
	)

if(UNIX)
	set(EXTRA_LIBS
		#z
		#ncurses
		tinfo
		pthread
		stdc++fs
		dl
		)
elseif(WIN32 OR CYGWIN)
	set(EXTRA_LIBS
		Version)
endif()

target_link_libraries(google-cpp-coding-style-checker
	PRIVATE
	${CHECK_ACTION_LIBS}
	# TODO: we don't know what the exact libraries dependencies, so hack here
	${CLANG_LIBS} ${CLANG_LIBS} ${CLANG_LIBS}
	${LLVM_LINK_COMPONENTS}
	${EXTRA_LIBS}
	)
