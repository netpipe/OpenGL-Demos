cmake_minimum_required(VERSION 3.10)
project(raytracer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(raytracer raytracer2.cpp)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 CONFIG QUIET)
if(NOT TARGET glfw AND NOT TARGET glfw3)
  find_package(glfw3 REQUIRED)
endif()

target_link_libraries(raytracer PRIVATE OpenGL::GL GLEW::GLEW)
if(TARGET glfw)
  target_link_libraries(raytracer PRIVATE glfw)
elseif(TARGET glfw3)
  target_link_libraries(raytracer PRIVATE glfw3)
else()
  target_link_libraries(raytracer PRIVATE glfw)
endif()

if(WIN32)
  target_compile_definitions(raytracer PRIVATE GLEW_STATIC)
endif()
