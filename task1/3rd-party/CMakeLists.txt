project(3rd-party)

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(3rd-party INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
