cmake_minimum_required(VERSION 3.20)
project(ml LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(BLAS REQUIRED)          # built-in CMake finder, no config file needed

add_library(ml)
target_sources(ml
  PRIVATE
    src/core/Matrix.cpp
    src/nn/model.cpp
    src/utils/utility.cpp
)
target_include_directories(ml
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(ml
  PUBLIC BLAS::BLAS                  # built-in target, always exists
)

add_executable(train main.cpp)
target_link_libraries(train PRIVATE ml)
