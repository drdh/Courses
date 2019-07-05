add_library(naming-check STATIC
    NamingCheck.cpp
    AdaptiveDict.cpp
    )
set(DICT_DIRECTORY "${CMAKE_BINARY_DIR}/${CCHECK_RESOURCE_DIR}/")
file(COPY "dicts" DESTINATION ${DICT_DIRECTORY})

install(DIRECTORY "dicts" DESTINATION "${CMAKE_INSTALL_PREFIX}/${CCHECK_RESOURCE_DIR}")