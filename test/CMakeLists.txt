enable_testing()

add_executable(boltimg_test test_conv.cpp)
target_link_libraries(boltimg_test boltimg GTest::gtest_main)
target_compile_definitions(boltimg_test PRIVATE -DBOLTIMG_TEST_ASSETS="${CMAKE_CURRENT_SOURCE_DIR}/assets/")

include(GoogleTest)
gtest_discover_tests(boltimg_test)