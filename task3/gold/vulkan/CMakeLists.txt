cmake_minimum_required(VERSION 3.16)
project(vk_noise C)

set(CMAKE_C_STANDARD 11)

# ── Vulkan ────────────────────────────────────────────────────────────────────
find_package(Vulkan REQUIRED)

# ── libvk_noise.so ────────────────────────────────────────────────────────────
add_library(vk_noise SHARED src/vk_noise.c)

target_include_directories(vk_noise PRIVATE
    src/
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(vk_noise PRIVATE
    ${Vulkan_LIBRARIES}
)

target_compile_options(vk_noise PRIVATE
    -O2 -Wall -Wextra
)

set_target_properties(vk_noise PROPERTIES
    OUTPUT_NAME             vk_noise
)

