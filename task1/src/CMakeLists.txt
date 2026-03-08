project(src)

find_package(Vulkan REQUIRED)

add_executable(task1 Main.cpp)

include_directories(${Vulkan_INCLUDE_DIR})

target_link_libraries(task1 ${ALL_LIBS} 3rd-party)

add_custom_command(TARGET task1 POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:task1> "${CMAKE_SOURCE_DIR}")