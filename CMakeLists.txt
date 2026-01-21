
cmake_minimum_required(VERSION 3.20)
project(ml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

add_executable(train main.cpp)

target_link_libraries(train PRIVATE ml)
