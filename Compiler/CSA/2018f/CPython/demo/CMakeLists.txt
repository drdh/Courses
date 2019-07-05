project(testcase)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS " -pthread ")
set(CMAKE_BUILD_TYPE Debug)
include_directories(./DebugInfo/include)
link_libraries(rt gvc cgraph cdt)

SET(EXECUTABLE_OUTPUT_PATH ./bin)
SET(LIBRARY_OUTPUT_PATH ./lib) 

add_library(wrapper SHARED
	./DebugInfo/wrapper.cpp
	./DebugInfo/Drawer.cpp
        ./DebugInfo/Global_Data.cpp
        ./DebugInfo/Function.cpp
        ./DebugInfo/Call.cpp
        ./DebugInfo/Sync.cpp)


add_executable(trysave
        ./src/trysave.cpp
        ./DebugInfo/Drawer.cpp
        ./DebugInfo/Global_Data.cpp
        ./DebugInfo/Function.cpp
        ./DebugInfo/Call.cpp
        ./DebugInfo/Sync.cpp)

add_executable(tryload
        ./src/tryload.cpp
        ./DebugInfo/Drawer.cpp
        ./DebugInfo/Global_Data.cpp
        ./DebugInfo/Function.cpp
        ./DebugInfo/Call.cpp
        ./DebugInfo/Sync.cpp)

add_executable(tryprint
        ./src/tryprint.cpp
        ./DebugInfo/Drawer.cpp
        ./DebugInfo/Global_Data.cpp
        ./DebugInfo/Function.cpp
        ./DebugInfo/Call.cpp
        ./DebugInfo/Sync.cpp)


