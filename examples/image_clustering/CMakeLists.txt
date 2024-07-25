add_executable(image_clustering main.cpp)

target_link_libraries(image_clustering PRIVATE ${THIRD_PARTY_LIBRARIES} imgui)
target_include_directories(image_clustering PRIVATE ${THIRD_PARTY_INCLUDE_DIRECTORIES} ${AI_INCLUDE_DIRECTORIES})