add_library(boltimg boltimg.c boltimg.h
boltimg_impl_avx2.c
boltimg_impl_avx512.c
boltimg_impl_sse2.c
boltimg_impl_sse4.c
boltimg_impl_scalar.c)

target_include_directories(boltimg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})