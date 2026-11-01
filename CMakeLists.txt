
cmake_minimum_required(VERSION 3.20)
project(labs VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(core STATIC
    src/dataset.cpp
    src/matrix.cpp
    src/matrix_slice.cpp
    src/decision_stump.cpp
    src/split_finder.cpp
    src/cart_tree.cpp
    src/cart_classifier.cpp
    src/cart_regressor.cpp
)

target_include_directories(core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(core PRIVATE -Wall -Wextra)

# Лабораторная №1
add_executable(app
    src/main.cpp
)

target_link_libraries(app PRIVATE core)
target_compile_options(app PRIVATE -Wall -Wextra)
target_include_directories(app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# Лабораторная №2
foreach(target lab2 lab2_gini lab2_variance)
    add_executable(${target} src/lab2_main.cpp)
    target_link_libraries(${target} PRIVATE core)
    target_compile_options(${target} PRIVATE -Wall -Wextra)
endforeach()

target_compile_definitions(lab2_variance PRIVATE LAB2_VARIANCE)

# Лабораторная №3
add_executable(lab3 src/lab3_main.cpp)
target_link_libraries(lab3 PRIVATE core)
target_compile_options(lab3 PRIVATE -Wall -Wextra)

# Относительный путь data/data.csv разрешается от корня проекта.
add_custom_target(run_lab3
    COMMAND $<TARGET_FILE:lab3>
    DEPENDS lab3
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    USES_TERMINAL
    VERBATIM
)

# Тесты
add_executable(tests
    tests/tests.cpp
)

target_link_libraries(tests PRIVATE core)
target_compile_options(tests PRIVATE -Wall -Wextra)

add_executable(test_decision_stump tests/test_decision_stump.cpp)
target_link_libraries(test_decision_stump PRIVATE core)
target_compile_options(test_decision_stump PRIVATE -Wall -Wextra)

add_executable(test_cart tests/test_cart.cpp)
target_link_libraries(test_cart PRIVATE core)
target_compile_options(test_cart PRIVATE -Wall -Wextra)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tests PRIVATE -UNDEBUG)
    target_compile_options(test_decision_stump PRIVATE -UNDEBUG)
    target_compile_options(test_cart PRIVATE -UNDEBUG)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    foreach(target core app lab2 lab2_gini lab2_variance lab3 tests test_decision_stump test_cart)
        target_compile_options(${target} PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
        target_link_options(${target} PRIVATE
            -fsanitize=address,undefined
        )
    endforeach()
endif()

enable_testing()
add_test(NAME unit COMMAND tests)
add_test(NAME decision_stump COMMAND test_decision_stump)
add_test(NAME cart COMMAND test_cart)

# Документация не входит в обычную сборку и не требует Graphviz.
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    configure_file(Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(docs
        COMMAND Doxygen::doxygen ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating HTML documentation with Doxygen"
        VERBATIM
    )
else()
    message(STATUS "Doxygen not found: install it with 'sudo apt install doxygen' to build HTML documentation; app, lab2 and tests remain available.")
    add_custom_target(docs
        COMMAND ${CMAKE_COMMAND} -E echo "Doxygen is required for docs. Install: sudo apt install doxygen. Then configure CMake again."
        COMMAND ${CMAKE_COMMAND} -E false
        VERBATIM
    )
endif()
