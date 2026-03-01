cmake_minimum_required(VERSION 3.20)
project(vecgraphdb VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(VECGRAPHDB_BUILD_PYTHON "Build Python bindings" ON)

include(FetchContent)

# hnswlib (header-only)
FetchContent_Declare(
  hnswlib
  GIT_REPOSITORY https://github.com/nmslib/hnswlib.git
  GIT_TAG v0.8.0
)
FetchContent_MakeAvailable(hnswlib)

# pybind11 (for Python bindings)
if(VECGRAPHDB_BUILD_PYTHON)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
  )
  FetchContent_MakeAvailable(pybind11)
endif()

add_library(vecgraphdb STATIC
  core/db_impl.cpp
)
set_target_properties(vecgraphdb PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(vecgraphdb PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${hnswlib_SOURCE_DIR}
)

# Python module
if(VECGRAPHDB_BUILD_PYTHON)
  pybind11_add_module(_vecgraphdb
    bindings/py_vecgraphdb.cpp
  )
  target_link_libraries(_vecgraphdb PRIVATE vecgraphdb)
  target_compile_definitions(_vecgraphdb PRIVATE VECGRAPHDB_VERSION=\"${PROJECT_VERSION}\")
endif()

# Tests (doctest header-only)
enable_testing()
add_executable(vecgraphdb_tests tests/test_basic.cpp)
target_include_directories(vecgraphdb_tests PRIVATE include)
target_link_libraries(vecgraphdb_tests PRIVATE vecgraphdb)
add_test(NAME vecgraphdb_tests COMMAND vecgraphdb_tests)
