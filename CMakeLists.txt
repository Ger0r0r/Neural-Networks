cmake_minimum_required(VERSION 3.13)

set(PROJECT_NAME NN_WORK)

project(${PROJECT_NAME} VERSION 1.0 LANGUAGES CXX C)

add_link_options(-static -static-libgcc -static-libstdc++)

include_directories(headers)

file(GLOB CPPS "src/*.cpp")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O2 -mavx2 -fopenmp")
message(${CMAKE_CXX_FLAGS_RELEASE})

add_executable(${PROJECT_NAME} ${CPPS})
target_link_libraries(${PROJECT_NAME} m)