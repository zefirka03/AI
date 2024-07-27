cmake_minimum_required(VERSION 3.15)
project(AI C CXX)

set(CMAKE_CXX_STANDARD 20)

set(glew-cmake_BUILD_SHARED FALSE)
add_definitions(-DGLEW_STATIC)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/third-party/glfw" glfw)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/third-party/glew-cmake" glew)

set(THIRD_PARTY_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/third-party/glfw/include"
                                    "${CMAKE_CURRENT_SOURCE_DIR}/third-party/glew-cmake/include"
                                    "${CMAKE_CURRENT_SOURCE_DIR}/third-party/imgui"
                                    "${CMAKE_CURRENT_SOURCE_DIR}/third-party/linalg")
set(THIRD_PARTY_LIBRARIES glfw libglew_static)
set(AI_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(GLOB imgui
     "third-party/imgui/*.cpp"
)

add_library(imgui ${imgui})

target_link_libraries(imgui PRIVATE ${THIRD_PARTY_LIBRARIES})
target_include_directories(imgui PRIVATE ${THIRD_PARTY_INCLUDE_DIRECTORIES})

add_subdirectory(examples)