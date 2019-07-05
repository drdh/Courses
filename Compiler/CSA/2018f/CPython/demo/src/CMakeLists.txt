project(testcase)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wl,--no-as-needed -Wl,--no-undefined  -fPIC")
set(CMAKE_BUILD_TYPE Debug)
SET(LIBRARY_OUTPUT_PATH ../lib) 
include_directories(../DebugInfo/include)
link_libraries(rt gvc cgraph cdt dl)

include_directories(../DebugInfo)
include_directories(../DebugInfo/include)
LINK_DIRECTORIES(../lib)
link_libraries(wrapper)
link_libraries(eculid)
add_library(hook SHARED
        hook.cpp
	hook.h)

