add_library(error-handling-check STATIC ErrorHandlingCheck.cpp)
configure_file("fName.txt" "${CMAKE_BINARY_DIR}/${CCHECK_RESOURCE_DIR}" COPYONLY)

install(FILES "fName.txt" DESTINATION "${CMAKE_INSTALL_PREFIX}/${CCHECK_RESOURCE_DIR}")