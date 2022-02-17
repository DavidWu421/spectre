# Distributed under the MIT License.
# See LICENSE.txt for details.

# Every time we've upgraded blaze compatibility in the past, we've had to change
# vector code, so we should expect to need changes again on each subsequent
# release, so we specify an exact version requirement.
find_package(Blaze 3.8 EXACT REQUIRED)

message(STATUS "Blaze incl: ${BLAZE_INCLUDE_DIR}")
message(STATUS "Blaze vers: ${BLAZE_VERSION}")

file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
  "Blaze version: ${BLAZE_VERSION}\n"
  )

add_library(Blaze INTERFACE IMPORTED)
set_property(TARGET Blaze PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES ${BLAZE_INCLUDE_DIR})
set_property(TARGET Blaze PROPERTY
  INTERFACE_LINK_LIBRARIES Lapack)
target_link_libraries(
  Blaze
  INTERFACE
  Blas
  GSL::gsl # for BLAS header
  Lapack
  )

# Configure Blaze. Some of the Blaze configuration options could be optimized
# for the machine we are running on. See documentation:
# https://bitbucket.org/blaze-lib/blaze/wiki/Configuration%20and%20Installation#!step-2-configuration
target_compile_definitions(Blaze
  INTERFACE
  # - Enable external BLAS kernels
  BLAZE_BLAS_MODE=1
  # - Use BLAS header from GSL. We could also find and include a <cblas.h> (or
  #   similarly named) header that may be distributed with the BLAS
  #   implementation, but it's not guaranteed to be available and may conflict
  #   with the GSL header. Since we use GSL anyway, it's easier to use their
  #   BLAS header.
  BLAZE_BLAS_INCLUDE_FILE=<gsl/gsl_cblas.h>
  # - Set default matrix storage order to column-major, since many of our
  #   functions are implemented for column-major layout. This default reduces
  #   conversions.
  BLAZE_DEFAULT_STORAGE_ORDER=blaze::columnMajor
  # - Disable SMP parallelization. This disables SMP parallelization for all
  #   possible backends (OpenMP, C++11 threads, Boost, HPX):
  #   https://bitbucket.org/blaze-lib/blaze/wiki/Serial%20Execution#!option-3-deactivation-of-parallel-execution
  BLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0
  # - Disable MPI parallelization
  BLAZE_MPI_PARALLEL_MODE=0
  # - Using the default cache size, which may have been configured automatically
  #   by the Blaze CMake configuration for the machine we are running on. We
  #   could override it here explicitly to tune performance.
  # BLAZE_CACHE_SIZE
  BLAZE_USE_PADDING=0
  # Enable non-temporal stores for cache optimization of large data structures
  BLAZE_USE_STREAMING=1
  BLAZE_USE_OPTIMIZED_KERNELS=1
  # Skip initializing default-constructed structures for fundamental types
  BLAZE_USE_DEFAULT_INITIALIZATON=0
  )

add_interface_lib_headers(
  TARGET Blaze
  HEADERS
  blaze/math/CustomVector.h
  blaze/math/DynamicMatrix.h
  blaze/math/DynamicVector.h
  blaze/system/Optimizations.h
  blaze/system/Version.h
  blaze/util/typetraits/RemoveConst.h
  )

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  Blaze
  )
