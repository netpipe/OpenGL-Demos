cmake_minimum_required(VERSION 3.10)
project(ocp_logo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(app test.cpp)

# test.cpp includes only GL/glew.h and GLFW/glfw3.h. GLUT and libpng used to be
# required here and are not used by anything, which made this fail to configure
# on machines that only have GLEW and GLFW.
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

find_package(glfw3 CONFIG QUIET)
if(NOT TARGET glfw AND NOT TARGET glfw3)
  find_package(glfw3 REQUIRED)
endif()

target_link_libraries(app PRIVATE OpenGL::GL GLEW::GLEW)

if(TARGET glfw)
  target_link_libraries(app PRIVATE glfw)
else()
  target_link_libraries(app PRIVATE glfw3)
endif()

if(WIN32)
  target_compile_definitions(app PRIVATE GLEW_STATIC)
endif()
