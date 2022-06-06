// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>

#include "DataStructures/CachedTempBuffer.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"

#include "Framework/Pypp.hpp"
#include "Framework/PyppFundamentals.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

#include "Helpers/PointwiseFunctions/AnalyticSolutions/GeneralRelativity/VerifyGrSolution.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <typeinfo>
#include <unordered_map>

// IWYU pragma: no_forward_declare Tags::deriv

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> spatial_coords(const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  const double dx_i = .0001;
  get<0>(x)[0] = 4;
  get<0>(x)[1] = 4 + dx_i;
  get<0>(x)[2] = 4;
  get<0>(x)[3] = 4;
  get<1>(x)[0] = 3;
  get<1>(x)[1] = 3;
  get<1>(x)[2] = 3 + dx_i;
  get<1>(x)[3] = 3;
  get<2>(x)[0] = 2;
  get<2>(x)[1] = 2;
  get<2>(x)[2] = 2;
  get<2>(x)[3] = 2 + dx_i;
  return x;
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.KerrSchild",
                  "[PointwiseFunctions][Unit]") {
  // Parameters for SphKerrSchild solution
  // Set up DataVector with same lenghts
  const DataVector used_for_size(4);

  const size_t used_for_sizet = used_for_size.size();
  const double mass = 0.5;
  const std::array<double, 3> spin{{0.1, -0.3, 0.2}};
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};

  // non perturbed spatial coordinates
  const auto x = spatial_coords<Frame::Inertial>(used_for_size);

  std::cout << "This is x:" << x << "\n";
  std::cout << "This is used for size:" << used_for_sizet << "\n";

  // Set up the solution, computer object, and cache object
  gr::Solutions::KerrSchild solution(mass, spin, center);

  // non perturbed computer
  gr::Solutions::KerrSchild::IntermediateComputer ks_computer(solution, x);
  gr::Solutions::KerrSchild::IntermediateVars<DataVector, Frame::Inertial>
      cache(used_for_sizet);

  std::cout << "Hello World"
            << "\n";

  // DERIV_L TEST
  // ONLY CALCULATES THE SPATIAL DERIVATIVES SINCE NO TIME EVOLUTION

  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/");
  Approx finite_difference_approx = Approx::custom().epsilon(1e-6).scale(1.0);

  const tnsr::i<DataVector, 3, Frame::Inertial>& pert_coords_wrong_type_l =
      cache.get_var(ks_computer,
                    gr::Solutions::KerrSchild::internal_tags::null_form<
                        DataVector, Frame::Inertial>{});

  std::cout << "This is null form" << pert_coords_wrong_type_l << "\n";

  tnsr::Ij<DataVector, 3, Frame::Inertial> pert_coords_right_type_l{1_st, 0.};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      pert_coords_right_type_l.get(i, j) = pert_coords_wrong_type_l[i][j + 1];
    }
  }

  auto input_coords_l =
      make_with_value<tnsr::I<double, 3, Frame::Inertial>>(1, 0.0);
  input_coords_l[0] = pert_coords_wrong_type_l[0][0];
  input_coords_l[1] = pert_coords_wrong_type_l[1][0];
  input_coords_l[2] = pert_coords_wrong_type_l[2][0];

  std::cout << "This is input_coords_l:"
            << "\n"
            << input_coords_l << "\n"
            << "This is pert_coords_right_type_l" << pert_coords_right_type_l
            << "\n";

  auto perturbation_l =
      make_with_value<tnsr::I<double, 3, Frame::Inertial>>(1, 0.0001);
  const auto finite_diff_deriv_l =
      pypp::call<tnsr::Ij<DataVector, 3, Frame::Inertial>>(
          "General_Finite_Difference", "check_finite_difference_rank1",
          input_coords_l, pert_coords_right_type_l, perturbation_l);

  tnsr::Ij<DataVector, 3, Frame::Inertial> input_coords_deriv_l{1_st, 0.};

  tnsr::ij<DataVector, 3, Frame::Inertial> deriv_null_form{used_for_size};
  ks_computer(make_not_null(&deriv_null_form), make_not_null(&cache),
              gr::Solutions::KerrSchild::internal_tags::deriv_null_form<
                  DataVector, Frame::Inertial>{});

  std::cout << "This is finite diff_null_form: \n"
            << finite_diff_deriv_l << "\n";
  std::cout << "This is deriv_null_form: \n" << deriv_null_form << "\n";


  tnsr::i<DataVector, 3, Frame::Inertial> null_form{used_for_size};
  ks_computer(make_not_null(&null_form), make_not_null(&cache),
               gr::Solutions::KerrSchild::internal_tags::null_form<
                   DataVector, Frame::Inertial>{});

    std::cout << "This is null_form: "
              << "\n"
              << std::setprecision(12) << null_form << "\n";

  Scalar<DataVector> lapse_squared{used_for_size};
  ks_computer(make_not_null(&lapse_squared), make_not_null(&cache),
              gr::Solutions::KerrSchild::internal_tags::lapse_squared<
                  DataVector>{});

  std::cout << "This is lapse_squared: \n" << lapse_squared << "\n";
}
