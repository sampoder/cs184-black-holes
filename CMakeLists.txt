cmake_minimum_required(VERSION 3.16)
project(cs184-black-holes)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Aggressive perf flags for the hot path (geodesic integrator).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-O3 -ffast-math -fno-math-errno -march=native -funroll-loops)
endif()

set(OpenGL_GL_PREFERENCE LEGACY)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(blackhole main.cpp)
target_link_libraries(blackhole PRIVATE glfw OpenGL::GL)

if(APPLE)
    target_link_libraries(blackhole PRIVATE "-framework Cocoa" "-framework IOKit" "-framework CoreVideo")
endif()
