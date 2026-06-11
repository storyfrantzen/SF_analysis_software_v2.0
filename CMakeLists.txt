cmake_minimum_required(VERSION 3.10)
project(Hipo2Root)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROOT setup
find_package(ROOT REQUIRED)
include(${ROOT_USE_FILE})

# Environment variable paths
set(CLAS12ROOT $ENV{CLAS12ROOT})
set(HIPO $ENV{HIPO})
set(CCDB_HOME $ENV{CCDB_HOME})
set(RCDB_HOME $ENV{RCDB_HOME})
set(QADB $ENV{QADB})

# Include directories
include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${CLAS12ROOT}/hipo4
  ${CLAS12ROOT}/Clas12Banks
  ${CLAS12ROOT}/Clas12Root
  ${HIPO}/include
  ${CCDB_HOME}/include
  ${RCDB_HOME}/cpp/include
  ${QADB}/srcC/include
)

# Link directories
link_directories(
  ${CLAS12ROOT}/lib64
  ${HIPO}/lib
  ${CCDB_HOME}/lib
)

# Collect all .cpp files
file(GLOB ALL_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Add the ROOT dictionary
set(ROOTBRANCHES_HEADERS ${PROJECT_SOURCE_DIR}/include/ROOTBranches.h)
ROOT_GENERATE_DICTIONARY(G__ROOTBranchesDict ${ROOTBRANCHES_HEADERS} LINKDEF ${PROJECT_SOURCE_DIR}/include/ROOTBranchesLinkDef.h)
list(APPEND ALL_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/G__ROOTBranchesDict.cxx)

# --- Library ---
add_library(ROOTBranchesDict SHARED ${CMAKE_CURRENT_BINARY_DIR}/G__ROOTBranchesDict.cxx)
target_include_directories(ROOTBranchesDict PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ROOTBranchesDict PRIVATE ROOT::Core ROOT::RIO ROOT::Net ROOT::Hist ROOT::Tree ROOT::Physics ROOT::EG)

install(TARGETS ROOTBranchesDict LIBRARY DESTINATION lib)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libROOTBranchesDict_rdict.pcm DESTINATION lib)

# --- Executable 1: processHipoFiles ---
add_executable(hipo2root
  ${PROJECT_SOURCE_DIR}/src/hipo2root.cpp
  ${ALL_SOURCES}   # shared code only
)
target_link_libraries(hipo2root
  ${ROOT_LIBRARIES}
  Clas12Root
  Clas12Banks
  hipo4
  EG
  pthread
)