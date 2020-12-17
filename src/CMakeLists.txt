cmake_minimum_required(VERSION 3.0)
project(MarchingCubes LANGUAGES CXX CUDA)

find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
add_library(${PROJECT_NAME} SHARED "")

target_sources(${PROJECT_NAME}
PRIVATE
MarchingCube/MarchingCube.cu)

target_include_directories(${PROJECT_NAME}
PUBLIC
${CMAKE_SOURCE_DIR}/MarchingCube)

target_link_libraries(${PROJECT_NAME}
PUBLIC
Eigen3::Eigen
Boost::program_options)

set_target_properties(${PROJECT_NAME}
PROPERTIES 
CUDA_SEPARABLE_COMPILATION ON
POSITION_INDEPENDENT_CODE ON
CUDA_STANDARD 11
CUDA_ARCHITECTURES 86)

add_executable(main Examples/main.cpp)
target_link_libraries(main ${PROJECT_NAME})