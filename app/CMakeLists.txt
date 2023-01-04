# Find GTK4
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK4 REQUIRED gtk4)
find_program(GLIB_COMPILE_RESOURCES NAMES glib-compile-resources REQUIRED)

set(GRESOURCE_C   ${CMAKE_CURRENT_BINARY_DIR}/bolty.gresource.c)
set(GRESOURCE_XML bolty.gresource.xml)

add_custom_command(
    OUTPUT ${GRESOURCE_C}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${GLIB_COMPILE_RESOURCES}
    ARGS
        --target=${GRESOURCE_C}
        ${GRESOURCE_XML}
    VERBATIM
    MAIN_DEPENDENCY ${GRESOURCE_XML}
    DEPENDS
        bolty-window.ui
)

set_source_files_properties(
    ${GRESOURCE_C}
    PROPERTIES GENERATED TRUE
)

add_executable(bolty bolty_main.c bolty_image.c bolty_image.h ${GRESOURCE_C})

target_link_libraries(bolty boltimg ${GTK4_LIBRARIES} m)
target_include_directories(bolty PRIVATE ${GTK4_INCLUDE_DIRS})
target_compile_options(bolty PRIVATE ${GTK4_CFLAGS_OTHER})
target_link_directories(bolty PRIVATE ${GTK4_LIBRARY_DIRS})