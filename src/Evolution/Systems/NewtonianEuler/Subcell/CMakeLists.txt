# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  InitialDataTci.cpp
  PrimitiveGhostData.cpp
  PrimsAfterRollback.cpp
  ResizeAndComputePrimitives.cpp
  TciOnDgGrid.cpp
  TciOnFdGrid.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ComputeFluxes.hpp
  InitialDataTci.hpp
  NeighborPackagedData.hpp
  PrimitiveGhostData.hpp
  PrimsAfterRollback.hpp
  ResizeAndComputePrimitives.hpp
  Subcell.hpp
  TciOnDgGrid.hpp
  TciOnFdGrid.hpp
  TimeDerivative.hpp
  )
