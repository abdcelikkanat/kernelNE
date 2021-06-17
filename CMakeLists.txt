cmake_minimum_required(VERSION 3.10)
project(kernelNE)

set(CMAKE_CXX_STANDARD 14)

include_directories(lib)

#add_subdirectory(ext/Graph)

#set(CMAKE_MODULE_PATH "./ext/Graph;${CMAKE_MODULE_PATH}")
set(GML_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext/Graph/src)
set(GML_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext/Graph/lib)
set(LPSOLVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ext/lpsolve55")

include_directories(${GML_LIB_DIR} ${LPSOLVE_DIR})

set(SOURCE_FILES src/Model.cpp src/Unigram.cpp src/Vocabulary.cpp src/Node.cpp)
set(LIB_FILES lib/Model.h lib/Unigram.h lib/Vocabulary.h lib/Node.h lib/Utilities.h)

add_executable(${CMAKE_PROJECT_NAME} main.cpp ${SOURCE_FILES} ${LIB_FILES})


#target_link_libraries(kernelNE PUBLIC ${GML_LIB_DIR} -L"${LPSOLVE_DIR}" -Wl,-rpath,"${LPSOLVE_DIR}" lpsolve55)
target_link_libraries(kernelNE PUBLIC ${GML_LIB_DIR})