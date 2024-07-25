add_executable(sin_approximation main.cpp)

target_link_libraries(sin_approximation PRIVATE ${THIRD_PARTY_LIBRARIES} imgui)
target_include_directories(sin_approximation PRIVATE ${THIRD_PARTY_INCLUDE_DIRECTORIES} ${AI_INCLUDE_DIRECTORIES})