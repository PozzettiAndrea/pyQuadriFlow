cmake_minimum_required(VERSION 3.15...3.26)
project(pyquadriflow LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Configure QuadriFlow as static library
# ---------------------------------------------------------------------------
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS FALSE)

# Disable optional QuadriFlow features we don't need
set(BUILD_PERFORMANCE_TEST OFF CACHE BOOL "" FORCE)
set(BUILD_LOG OFF CACHE BOOL "" FORCE)
set(BUILD_GUROBI OFF CACHE BOOL "" FORCE)
set(BUILD_TBB OFF CACHE BOOL "" FORCE)
set(BUILD_OPENMP OFF CACHE BOOL "" FORCE)
set(BUILD_FREE_LICENSE ON CACHE BOOL "" FORCE)

if(WIN32)
  add_definitions(/D_USE_MATH_DEFINES /DNOMINMAX)
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)
endif()

# Build QuadriFlow — EXCLUDE_FROM_ALL suppresses its install rules and exe target
add_subdirectory(quadriflow EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# Python + nanobind
# ---------------------------------------------------------------------------
find_package(Python 3.10
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

find_package(nanobind CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# Static library for QuadriFlow pipeline (pure C++ — NO Python headers)
# ---------------------------------------------------------------------------
add_library(quadriflow_pipeline STATIC
  src/pipeline.cpp
)

target_link_libraries(quadriflow_pipeline PUBLIC quadriflow)

target_include_directories(quadriflow_pipeline PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  # QuadriFlow headers
  ${CMAKE_CURRENT_SOURCE_DIR}/quadriflow/src
  ${CMAKE_CURRENT_SOURCE_DIR}/quadriflow/3rd/pcg32
  ${CMAKE_CURRENT_SOURCE_DIR}/quadriflow/3rd/pss
)

# ---------------------------------------------------------------------------
# Nanobind extension module (Python + nanobind ONLY — no QuadriFlow headers)
# ---------------------------------------------------------------------------
nanobind_add_module(
  _pyquadriflow
  STABLE_ABI
  src/_pyquadriflow.cpp
)

# Link the pipeline library — only pipeline.h is included, no QuadriFlow leaks
target_link_libraries(_pyquadriflow PRIVATE quadriflow_pipeline)

target_include_directories(_pyquadriflow PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Compiler-specific optimization
if(MSVC)
  target_compile_options(_pyquadriflow PRIVATE /O2)
  target_compile_options(quadriflow_pipeline PRIVATE /O2)
else()
  target_compile_options(_pyquadriflow PRIVATE -O3)
  target_compile_options(quadriflow_pipeline PRIVATE -O3)
endif()

# Install into the pyquadriflow Python package directory
install(TARGETS _pyquadriflow LIBRARY DESTINATION pyquadriflow)
