set(CCHECK_RESOURCE_DIR "share/ccheck")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/${CCHECK_RESOURCE_DIR}")
add_subdirectory(ErrorHandlingCheck)
add_subdirectory(FullCommentCheck)
add_subdirectory(HeaderCheck)
add_subdirectory(InitInNeedCheck)
add_subdirectory(ModuleCheck)
add_subdirectory(NamingCheck)

add_executable(code-spec-checker
	ToolDriver.cpp
	# StateStore.cpp
	)
install(TARGETS code-spec-checker DESTINATION "${CMAKE_INSTALL_PREFIX}/bin/")

set(CHECK_ACTION_LIBS
	error-handling-check
	full-comment-check
	header-check
	init-in-need-check;
	module-check
	naming-check
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
		z
		ncurses
		pthread
		stdc++fs
		dl
		)
elseif(WIN32 OR CYGWIN)
	set(EXTRA_LIBS
		Version)
endif()

target_link_libraries(code-spec-checker
	PRIVATE
	${CHECK_ACTION_LIBS}
	# TODO: we don't know what the exact libraries dependencies, so hack here
	${CLANG_LIBS} ${CLANG_LIBS} ${CLANG_LIBS}
	${LLVM_LINK_COMPONENTS}
	${EXTRA_LIBS}
	)