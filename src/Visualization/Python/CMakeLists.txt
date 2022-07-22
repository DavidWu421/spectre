# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyVisualization")

spectre_python_add_module(
  Visualization
  PYTHON_EXECUTABLES
  ExtendConnectivity.py
  GenerateXdmf.py
  InterpolateVolumeData.py
  Render1D.py
)
