# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DomainTimeDependence")

set(LIBRARY_SOURCES
  Test_CubicScale.cpp
  Test_None.cpp
  Test_RotationAboutZAxis.cpp
  Test_ScalingAndZRotation.cpp
  Test_Shape.cpp
  Test_SphericalCompression.cpp
  Test_UniformTranslation.cpp
  )

add_test_library(
  ${LIBRARY}
  "Domain/Creators/TimeDependence"
  "${LIBRARY_SOURCES}"
  "DataStructuresHelpers;DomainHelpers;DomainTimeDependence;Options"
  )
