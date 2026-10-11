cmake_minimum_required(VERSION 3.16)
project(scd_experiments LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
# Match the optimization settings used to generate the archived paper data.
# Native builds depend on the build machine CPU; disable for portable builds.
option(SCD_NATIVE_OPTIMIZATION "Use the original native CPU optimization flags" ON)
if(SCD_NATIVE_OPTIMIZATION AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-march=native -funroll-loops)
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
find_package(Threads REQUIRED)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(open_spiel/abseil-cpp EXCLUDE_FROM_ALL)
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/open_spiel/abseil-cpp)
set(SCD_LIBS absl::flags absl::flags_parse absl::flat_hash_map absl::flat_hash_set
    absl::node_hash_map absl::optional absl::random_random absl::str_format
    absl::strings absl::time Threads::Threads)
add_library(scd_core OBJECT
  hr_edl/best_response.cc
  hr_edl/decision_point.cc
  hr_edl/math.cc
  hr_edl/policy_evaluation.cc
  hr_edl/samplers.cc
  hr_edl/spiel_extra.cc
  open_spiel/action_view.cc
  open_spiel/game_parameters.cc
  open_spiel/game_transforms/turn_based_simultaneous_game.cc
  open_spiel/games/battleship.cc
  open_spiel/games/battleship_types.cc
  open_spiel/games/efg_game.cc
  open_spiel/games/goofspiel.cc
  open_spiel/games/kuhn_poker.cc
  open_spiel/games/leduc_poker.cc
  open_spiel/games/sheriff.cc
  open_spiel/matrix_game.cc
  open_spiel/observer.cc
  open_spiel/policy.cc
  open_spiel/simultaneous_move_game.cc
  open_spiel/spiel.cc
  open_spiel/spiel_utils.cc
  open_spiel/utils/file.cc
)
target_link_libraries(scd_core PUBLIC ${SCD_LIBS})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
foreach(name run_fixed_ltbr show_game_size show_max_and_min_utilities)
  add_executable(${name} bin/${name}.cc $<TARGET_OBJECTS:scd_core>)
  target_link_libraries(${name} PRIVATE ${SCD_LIBS})
endforeach()
option(SCD_BUILD_TESTS "Build the scientific implementation checks" ON)
if(SCD_BUILD_TESTS)
  enable_testing()
  foreach(name scd_local_coverage_test tabular_learner_test action_transformation_test)
    add_executable(${name} hr_edl/${name}.cc $<TARGET_OBJECTS:scd_core>)
    target_link_libraries(${name} PRIVATE ${SCD_LIBS})
    # Keep standard assertions enabled even in a Release build.
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(${name} PRIVATE -UNDEBUG)
    endif()
    add_test(NAME ${name} COMMAND ${name})
  endforeach()
endif()
