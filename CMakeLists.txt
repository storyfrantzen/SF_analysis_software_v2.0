cmake_minimum_required(VERSION 3.10)
project(Hipo2Root)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CTest)

# ROOT setup
find_package(ROOT REQUIRED)
include(${ROOT_USE_FILE})

# Environment variable paths
set(CLAS12ROOT $ENV{CLAS12ROOT})
set(HIPO $ENV{HIPO})
set(CCDB_HOME $ENV{CCDB_HOME})
set(RCDB_HOME $ENV{RCDB_HOME})

find_path(QADB_INCLUDE_DIR QADB.h
  HINTS $ENV{QADB}/srcC/include
)

# Include directories
include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/core
  ${PROJECT_SOURCE_DIR}/include/conversion
  ${PROJECT_SOURCE_DIR}/include/post
  ${PROJECT_SOURCE_DIR}/vendor
  ${CLAS12ROOT}/hipo4
  ${CLAS12ROOT}/Clas12Banks
  ${CLAS12ROOT}/Clas12Root
  ${HIPO}/include
  ${CCDB_HOME}/include
  ${RCDB_HOME}/cpp/include
)

# Link directories
link_directories(
  ${CLAS12ROOT}/lib64
  ${HIPO}/lib
  ${CCDB_HOME}/lib
)

set(HIPO2ROOT_SOURCES
  ${PROJECT_SOURCE_DIR}/src/conversion/GeneratedEvent.cpp
  ${PROJECT_SOURCE_DIR}/src/conversion/ProtonEnergyLossCorrections.cpp
  ${PROJECT_SOURCE_DIR}/src/conversion/QualityAssurance.cpp
  ${PROJECT_SOURCE_DIR}/src/core/Kinematics.cpp
  ${PROJECT_SOURCE_DIR}/src/core/ROOTBranches.cpp
)

set(POST_PROCESSING_SOURCES
  ${PROJECT_SOURCE_DIR}/src/core/Kinematics.cpp
  ${PROJECT_SOURCE_DIR}/src/post/Cuts.cpp
  ${PROJECT_SOURCE_DIR}/src/core/ROOTBranches.cpp
)

# Generate and compile the ROOT dictionary exactly once.
set(ROOTBRANCHES_HEADERS ${PROJECT_SOURCE_DIR}/include/core/ROOTBranches.h)
ROOT_GENERATE_DICTIONARY(
  G__ROOTBranchesDict
  ${ROOTBRANCHES_HEADERS}
  LINKDEF ${PROJECT_SOURCE_DIR}/include/core/ROOTBranchesLinkDef.h
)
add_library(
  ROOTBranchesDictObjects OBJECT
  ${CMAKE_CURRENT_BINARY_DIR}/G__ROOTBranchesDict.cxx
)
set_target_properties(ROOTBranchesDictObjects PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(
  ROOTBranchesDictObjects PRIVATE
  ${PROJECT_SOURCE_DIR}/include/core
)

# --- Library ---
add_library(
  ROOTBranchesDict SHARED
  $<TARGET_OBJECTS:ROOTBranchesDictObjects>
)
target_link_libraries(
  ROOTBranchesDict PRIVATE
  ROOT::Core ROOT::RIO ROOT::Net ROOT::Hist ROOT::Tree ROOT::Physics ROOT::EG
)

install(TARGETS ROOTBranchesDict LIBRARY DESTINATION lib)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libROOTBranchesDict_rdict.pcm DESTINATION lib)

# --- Executable 1: hipo2root ---
add_executable(
  hipo2root
  ${PROJECT_SOURCE_DIR}/src/apps/hipo2root.cpp
  ${HIPO2ROOT_SOURCES}
  $<TARGET_OBJECTS:ROOTBranchesDictObjects>
)
target_link_libraries(
  hipo2root
  ${ROOT_LIBRARIES}
  Clas12Root
  Clas12Banks
  hipo4
  EG
  pthread
)
if(QADB_INCLUDE_DIR)
  target_include_directories(hipo2root PRIVATE ${QADB_INCLUDE_DIR})
  target_compile_definitions(hipo2root PRIVATE HAVE_QADB=1)
  message(STATUS "QADB support enabled: ${QADB_INCLUDE_DIR}")
else()
  message(STATUS "QADB support disabled (QADB.h not found)")
endif()

# --- Executable 2: ROOT post-processing ---
add_executable(
  post_process
  ${PROJECT_SOURCE_DIR}/src/apps/post_process.cpp
  ${POST_PROCESSING_SOURCES}
  $<TARGET_OBJECTS:ROOTBranchesDictObjects>
)
target_compile_definitions(post_process PRIVATE POST_PROCESSING_ONLY)
target_link_libraries(
  post_process
  ${ROOT_LIBRARIES}
  EG
  pthread
)

if(BUILD_TESTING)
  add_executable(
    test_conversion_config
    ${PROJECT_SOURCE_DIR}/tests/test_conversion_config.cpp
  )
  add_test(NAME conversion_config COMMAND test_conversion_config)

  add_executable(
    test_post_cuts
    ${PROJECT_SOURCE_DIR}/tests/test_post_cuts.cpp
    ${POST_PROCESSING_SOURCES}
    $<TARGET_OBJECTS:ROOTBranchesDictObjects>
  )
  target_compile_definitions(test_post_cuts PRIVATE POST_PROCESSING_ONLY)
  target_link_libraries(
    test_post_cuts
    ${ROOT_LIBRARIES}
    EG
    pthread
  )
  add_test(NAME post_cuts COMMAND test_post_cuts)
endif()
