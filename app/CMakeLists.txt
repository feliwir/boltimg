find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK4 REQUIRED gtk4)

link_directories(${GTK4_LIBRARY_DIRS})

add_definitions(${GTK4_CFLAGS_OTHER})

add_executable(bolty app_main.c bolty_image.c bolty_image.h)

target_link_libraries(bolty boltimg ${GTK4_LIBRARIES} m)
target_include_directories(bolty PRIVATE ${GTK4_INCLUDE_DIRS})
