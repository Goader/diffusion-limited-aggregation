cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(diffusion_limited_aggregation LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)

# fixme this should not be here, but I couldn't make CLion to find nvcc
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA found")
else()
    message(STATUS "CUDA not found")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${OPENGL_INCLUDE_DIR} ${GLFW_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})

find_package(nlohmann_json 3.2.0 REQUIRED)  # json
find_package(OpenGL REQUIRED)  # OpenGL
find_package(glfw3 REQUIRED)  # GLFW - bare necessities for interaction with OpenGL
find_package(GLEW REQUIRED)  # GLEW - OpenGL + extensions loader

add_subdirectory(src)
add_executable(diffusion_limited_aggregation ${SOURCE} ${HEADERS})

target_link_libraries(
    diffusion_limited_aggregation PRIVATE ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} glfw nlohmann_json::nlohmann_json
    )
