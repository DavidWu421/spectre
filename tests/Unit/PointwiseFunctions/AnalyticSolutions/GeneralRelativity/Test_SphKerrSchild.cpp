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
#include "DataStructures/Tensor/EagerMath/CrossProduct.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/PyppFundamentals.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Framework/TestingFramework.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/GeneralRelativity/VerifyGrSolution.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Solutions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphKerrSchild.hpp"
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

// set up non perturbed spatial coordinates
template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> spatial_coords(const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  const double dx_i = .0001;
  get<0>(x)[0] = 4.;
  get<0>(x)[1] = 4. + dx_i;
  get<0>(x)[2] = 4.;
  get<0>(x)[3] = 4.;
  get<1>(x)[0] = 3.;
  get<1>(x)[1] = 3.;
  get<1>(x)[2] = 3. + dx_i;
  get<1>(x)[3] = 3.;
  get<2>(x)[0] = 2.;
  get<2>(x)[1] = 2.;
  get<2>(x)[2] = 2.;
  get<2>(x)[3] = 2. + dx_i;
  return x;
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.SphKerrSchild",
                  "[PointwiseFunctions][Unit]") {
  // Parameters for SphKerrSchild solution
  // Set up DataVector with same lenghts
  const DataVector used_for_size(4);

  const size_t used_for_sizet = used_for_size.size();
  const double mass = 0.5;
  const std::array<double, 3> spin{{0.1, -0.3, 0.2}};
  const std::array<double, 3> center{{0.5, 0.3, -2.0}};

  // non perturbed spatial coordinates
  const auto x = spatial_coords<Frame::Inertial>(used_for_size);

  // Set up the solution, computer object, and cache object
  gr::Solutions::SphKerrSchild solution(mass, spin, center);

  // non perturbed computer
  gr::Solutions::SphKerrSchild::IntermediateComputer sks_computer(solution, x);
  gr::Solutions::SphKerrSchild::IntermediateVars<DataVector, Frame::Inertial>
      cache(used_for_sizet);

  // Functions outputs and tests

  // r test - non perturbed
  Scalar<DataVector> r(3_st, 0.);
  sks_computer(make_not_null(&r), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::r<DataVector>{});

  //   std::cout << "This is r: "
  //             << "\n"
  //             << r << "\n";

  // rho test - non perturbed
  Scalar<DataVector> rho(3_st, 0.);
  sks_computer(make_not_null(&rho), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::rho<DataVector>{});

  //   std::cout << "This is rho: "
  //             << "\n"
  //             << rho << "\n";

  // jacobian test - non perturbed
  tnsr::Ij<DataVector, 3, Frame::Inertial> jacobian{1_st, 0.};
  sks_computer(
      make_not_null(&jacobian), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::jacobian<DataVector,
                                                            Frame::Inertial>{});

  // std::cout << "This is the jacobian: " << std::setprecision(16) << "\n"
  //           << jacobian << std::endl;

  // deriv_jacobian test
  tnsr::iJk<DataVector, 3, Frame::Inertial> deriv_jacobian{1_st, 0.};
  sks_computer(make_not_null(&deriv_jacobian), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::deriv_jacobian<
                   DataVector, Frame::Inertial>{});

  // std::cout << "This is the deriv_jacobian: "
  //           << "\n"
  //           << deriv_jacobian << std::endl;

  // inv_jacobian test - non perturbed
  tnsr::Ij<DataVector, 3, Frame::Inertial> inv_jacobian{1_st, 0.};
  sks_computer(make_not_null(&inv_jacobian), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::inv_jacobian<
                   DataVector, Frame::Inertial>{});

  //   std::cout << "This is inv_jacobian:" << "\n" << inv_jacobian << "\n";

  // matrix_E1 test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_E1{1_st, 0.};
  sks_computer(make_not_null(&matrix_E1), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::matrix_E1<
                   DataVector, Frame::Inertial>{});

  // matrix_E2 test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_E2{1_st, 0.};
  sks_computer(make_not_null(&matrix_E2), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::matrix_E2<
                   DataVector, Frame::Inertial>{});

  //   std::cout << "This is matrix E2: "
  //             << "\n"
  //             << matrix_E2 << "\n";

  //   deriv_inv_jacobian test - non perturbed
  tnsr::iJk<DataVector, 3, Frame::Inertial> deriv_inv_jacobian{1_st, 0.};
  sks_computer(make_not_null(&deriv_inv_jacobian), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::deriv_inv_jacobian<
                   DataVector, Frame::Inertial>{});

  // std::cout << "This is deriv inv jacobian: "
  //           << "\n"
  //           << deriv_inv_jacobian << "\n";

  tnsr::ijk<DataVector, 3, Frame::Inertial> deriv_jac_and_deriv_inv_jac_test{
      1_st, 0.};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t m = 0; m < 3; ++m) {
          for (size_t l = 0; l < 0; ++l) {
            deriv_jac_and_deriv_inv_jac_test.get(i, j, k) +=
                deriv_jacobian.get(i, j, k) + deriv_inv_jacobian.get(i, j, m) *
                                                  jacobian.get(m, l) *
                                                  jacobian.get(l, k);
          }
        }
      }
    }
  }
  // std::cout << "This is deriv_jac_and_deriv_inv_jac_test:"
  //           << "\n"
  //           << deriv_jac_and_deriv_inv_jac_test << "\n";

  //  H test - non perturbed
  Scalar<DataVector> H{0.};
  sks_computer(make_not_null(&H), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::H<DataVector>{});

  // std::cout << "This is scalar H: "
  //           << "\n"
  //           << std::setprecision(10) << H << "\n";

  // kerr_schild_l test - non perturbed
  auto kerr_schild_l = spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer(make_not_null(&kerr_schild_l), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::kerr_schild_l<
                   DataVector, Frame::Inertial>{});

  //   std::cout << "This is kerr_schild_l: "
  //             << "\n"
  //             << kerr_schild_l << "\n";

  // sph_kerr_schild_l_lower test - non perturbed
  tnsr::i<DataVector, 4, Frame::Inertial> sph_kerr_schild_l_lower{
      used_for_size};
  sks_computer(
      make_not_null(&sph_kerr_schild_l_lower), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::sph_kerr_schild_l_lower<
          DataVector, Frame::Inertial>{});

  std::cout << "This is sph_kerr_schild_l_lower: "
            << "\n"
            << std::setprecision(10) << sph_kerr_schild_l_lower << "\n";

  // sph_kerr_schild_l_upper test - non perturbed
  tnsr::I<DataVector, 4, Frame::Inertial> sph_kerr_schild_l_upper{
      used_for_size};
  sks_computer(
      make_not_null(&sph_kerr_schild_l_upper), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::sph_kerr_schild_l_upper<
          DataVector, Frame::Inertial>{});

  //   std::cout << "This is sph_kerr_schild_l_upper: "
  //             << "\n"
  //             << sph_kerr_schild_l_upper << "\n";

  // deriv_H test - non perturbed
  tnsr::I<DataVector, 4, Frame::Inertial> deriv_H{used_for_size};
  sks_computer(
      make_not_null(&deriv_H), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::deriv_H<DataVector,
                                                           Frame::Inertial>{});

  // std::cout << "This is deriv_H: "
  //           << "\n"
  //           << std::setprecision(10) << deriv_H << "\n";

  //   deriv_l - non perturbed
  tnsr::ij<DataVector, 4, Frame::Inertial> deriv_l{used_for_size};
  sks_computer(
      make_not_null(&deriv_l), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::deriv_l<DataVector,
                                                           Frame::Inertial>{});

  std::cout << "This is deriv_l: "
            << "\n"
            << deriv_l << "\n";

  Scalar<DataVector> lapse_squared{used_for_size};
  sks_computer(
      make_not_null(&lapse_squared), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::lapse_squared<DataVector>{});

  //   std::cout << "This is lapse_squared: \n" << lapse_squared << "\n";

  Scalar<DataVector> lapse{used_for_size};
  sks_computer(make_not_null(&lapse), make_not_null(&cache),
               gr::Tags::Lapse<DataVector>{});

  // std::cout << "This is lapse: \n" << std::setprecision(10) << lapse << "\n";

  tnsr::I<DataVector, 3, Frame::Inertial> shift{used_for_size};
  sks_computer(make_not_null(&shift), make_not_null(&cache),
               gr::Tags::Shift<3, Frame::Inertial, DataVector>{});

  // std::cout << "This is shift: "
  //           << "\n"
  //           << std::setprecision(10) << shift << "\n";

  tnsr::iJ<DataVector, 3, Frame::Inertial> deriv_shift{used_for_size};
  sks_computer(
      make_not_null(&deriv_shift), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::DerivShift<DataVector, Frame::Inertial>{});

  // std::cout << "This is deriv_shift: "
  //           << "\n"
  //           << std::setprecision(12) << deriv_shift << "\n";

  Scalar<DataVector> deriv_lapse_multiplier{used_for_size};
  sks_computer(
      make_not_null(&deriv_lapse_multiplier), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::deriv_lapse_multiplier<
          DataVector>{});

  // std::cout << "This is deriv_lapse_multiplier: \n" <<
  // deriv_lapse_multiplier << "\n";

  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{used_for_size};
  sks_computer(make_not_null(&spatial_metric), make_not_null(&cache),
               gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>{});

  // std::cout << "This is spatial_metric: "
  //           << "\n"
  //           << std::setprecision(12) << spatial_metric << "\n";

  tnsr::ijj<DataVector, 3, Frame::Inertial> deriv_spatial_metric{used_for_size};
  sks_computer(
      make_not_null(&deriv_spatial_metric), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::DerivSpatialMetric<DataVector,
                                                       Frame::Inertial>{});

  // std::cout << "This is deriv_spatial_metric: "
  //           << "\n"
  //           << std::setprecision(12) << deriv_spatial_metric << "\n";

  tnsr::ii<DataVector, 3, Frame::Inertial> dt_spatial_metric{used_for_size};
  sks_computer(
      make_not_null(&dt_spatial_metric), make_not_null(&cache),
      ::Tags::dt<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>{});

  //   std::cout << "This is dt_spatial_metric: "
  //             << "\n"
  //             << dt_spatial_metric << "\n";

  auto inv_spatial_metric =
      make_with_value<tnsr::II<DataVector, 3, Frame::Inertial>>(H, 0.);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {  // Symmetry
      inv_spatial_metric.get(i, j) -= 2.0 * get_element(H, 0) *
                                      get_element(lapse_squared, 0) *
                                      sph_kerr_schild_l_upper.get(i + 1) *
                                      sph_kerr_schild_l_upper.get(j + 1);
    }
  }

  for (size_t k = 0; k < 3; ++k) {
    for (size_t m = k; m < 3; ++m) {
      for (size_t i = 0; i < 3; ++i) {
        inv_spatial_metric.get(k, m) +=
            inv_jacobian.get(k, i) * inv_jacobian.get(m, i);
      }
    }
  }
  // std::cout << "This is inv_spatial_metric" << inv_spatial_metric << "\n";

  auto deriv_lapse =
      make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(H, 0.);
  for (size_t i = 0; i < 3; ++i) {
    deriv_lapse.get(i) = deriv_lapse_multiplier[0] * deriv_H[i + 1];
  }
  //   std::cout << "This is deriv_lapse" << std::setprecision(10) <<
  //   deriv_lapse
  //             << "\n";

  //   FINITE DIFFERENCE TESTS

  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/");
  Approx finite_difference_approx = Approx::custom().epsilon(1e-6).scale(1.0);

  // JACOBIAN TEST
  const tnsr::I<DataVector, 3, Frame::Inertial>& pert_coords_wrong_type =
      cache.get_var(sks_computer,
                    gr::Solutions::SphKerrSchild::internal_tags::x_kerr_schild<
                        DataVector, Frame::Inertial>{});
  tnsr::Ij<DataVector, 3, Frame::Inertial> pert_coords_right_type{1_st, 0.};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 1; j < 4; ++j) {
      pert_coords_right_type.get(i, j - 1) = pert_coords_wrong_type[i][j];
    }
  }

  auto input_coords =
      make_with_value<tnsr::I<double, 3, Frame::Inertial>>(1, 0.0);
  input_coords[0] = pert_coords_wrong_type[0][0];
  input_coords[1] = pert_coords_wrong_type[1][0];
  input_coords[2] = pert_coords_wrong_type[2][0];

  auto perturbation =
      make_with_value<tnsr::I<double, 3, Frame::Inertial>>(1, 0.0001);
  const auto finite_diff_jacobian =
      pypp::call<tnsr::Ij<DataVector, 3, Frame::Inertial>>(
          "General_Finite_Difference", "check_finite_difference_rank1",
          input_coords, pert_coords_right_type, perturbation);

  tnsr::Ij<DataVector, 3, Frame::Inertial> input_coords_jacobian{1_st, 0.};

  // Selects the jacobian for the input coords out of the jacobian matrix with 4
  // sets of coordiantes
  for (size_t i = 0; i < 9; i++) {
    input_coords_jacobian.get(i % 3, i / 3) = jacobian[i][0];
  }

  //   std::cout << "This is finite_diff_jac:" << "\n" << finite_diff_jacobian
  //   << "\n";

  //   CHECK_ITERABLE_CUSTOM_APPROX(finite_diff_jacobian, input_coords_jacobian,
  //                                finite_difference_approx);

  //   std::cout << "finite diff jacobian: " << finite_diff_jacobian << "\n";

  // DERIV JACOBIAN TEST

  const tnsr::Ij<DataVector, 3, Frame::Inertial>& pert_jacs_wrong_type =
      cache.get_var(sks_computer,
                    gr::Solutions::SphKerrSchild::internal_tags::jacobian<
                        DataVector, Frame::Inertial>{});
  auto input_jacs =
      make_with_value<tnsr::Ij<double, 3, Frame::Inertial>>(1, 0.0);
  auto pert_jacs_right_type =
      make_with_value<tnsr::ijk<double, 3, Frame::Inertial>>(1, 0.0);

  for (size_t l = 0; l < 9; ++l) {
    input_jacs[l] = pert_jacs_wrong_type[l][0];
  }

  size_t q = 0;
  size_t t = 0;
  for (size_t i = 0; i < 27; ++i) {
    pert_jacs_right_type[i] = pert_jacs_wrong_type[i % 9][t + 1];
    q += 1;
    if (q % 9 == 0) {
      t += 1;
    }
  }

  const auto finite_diff_deriv_jacobian =
      pypp::call<tnsr::iJk<DataVector, 3, Frame::Inertial>>(
          "General_Finite_Difference", "check_finite_difference_rank2",
          input_jacs, pert_jacs_right_type, perturbation);

  tnsr::iJk<DataVector, 3, Frame::Inertial> input_coords_deriv_jacobian{1_st,
                                                                        0.};

  // Selects the deriv jacobian for the input coords out of the deriv jacobian
  // matrix with 4 sets of coordiantes
  for (size_t i = 0; i < 27; i++) {
    input_coords_deriv_jacobian[i] = deriv_jacobian[i][0];
  }

  //   CHECK_ITERABLE_CUSTOM_APPROX(finite_diff_deriv_jacobian,
  //                                input_coords_deriv_jacobian,
  //                                finite_difference_approx);

  // DERIV_H TEST

  auto pert_coords_wrong_type_H = cache.get_var(
      sks_computer,
      gr::Solutions::SphKerrSchild::internal_tags::H<DataVector>{});
  tnsr::I<DataVector, 3, Frame::Inertial> pert_coords_right_type_H{1_st, 0.};
  for (size_t i = 0; i < 3; ++i) {
    pert_coords_right_type_H.get(i) = pert_coords_wrong_type_H[0][i + 1];
  }

  //   auto input_coords_H = make_with_value<Scalar<DataVector>>(1_st, 0.0);
  tnsr::I<DataVector, 1, Frame::Inertial> input_coords_H{1_st, 0.};
  input_coords_H[0] = pert_coords_wrong_type_H[0][0];

  const auto finite_diff_deriv_H =
      pypp::call<tnsr::I<DataVector, 3, Frame::Inertial>>(
          "General_Finite_Difference", "check_finite_difference_rank0",
          input_coords_H, pert_coords_right_type_H, perturbation);

  tnsr::I<DataVector, 3, Frame::Inertial> input_coords_deriv_H{1_st, 0.};
  for (size_t i = 0; i < 3; i++) {
    input_coords_deriv_H[i] = deriv_H[i + 1][0];
  }

  //   CHECK_ITERABLE_CUSTOM_APPROX(finite_diff_deriv_H, input_coords_deriv_H,
  //                                finite_difference_approx);

  // DERIV_L TEST
  // ONLY CALCULATES THE SPATIAL DERIVATIVES SINCE NO TIME EVOLUTION

  const tnsr::i<DataVector, 4, Frame::Inertial>& pert_coords_wrong_type_l =
      cache.get_var(
          sks_computer,
          gr::Solutions::SphKerrSchild::internal_tags::sph_kerr_schild_l_lower<
              DataVector, Frame::Inertial>{});
  tnsr::Ij<DataVector, 3, Frame::Inertial> pert_coords_right_type_l{1_st, 0.};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 1; j < 4; ++j) {
      pert_coords_right_type_l.get(i, j - 1) =
          pert_coords_wrong_type_l[i + 1][j];
    }
  }

  auto input_coords_l =
      make_with_value<tnsr::I<double, 3, Frame::Inertial>>(1, 0.0);
  input_coords_l[0] = pert_coords_wrong_type_l[1][0];
  input_coords_l[1] = pert_coords_wrong_type_l[2][0];
  input_coords_l[2] = pert_coords_wrong_type_l[3][0];

  auto perturbation_l =
      make_with_value<tnsr::I<double, 3, Frame::Inertial>>(1, 0.0001);
  const auto finite_diff_deriv_l =
      pypp::call<tnsr::Ij<DataVector, 3, Frame::Inertial>>(
          "General_Finite_Difference", "check_finite_difference_rank1",
          input_coords_l, pert_coords_right_type_l, perturbation_l);

  tnsr::Ij<DataVector, 3, Frame::Inertial> input_coords_deriv_l{1_st, 0.};

  // Selects the deriv_l for the input coords l out of the deriv_l matrix with 4
  // sets of coordiantes
  for (size_t i = 0; i < 12; i++) {
    if (i % 4 != 0) {
      input_coords_deriv_l.get((i - 1) % 4, (i - 1) / 4) = deriv_l[i + 4][0];
    }
  }

  //   CHECK_ITERABLE_CUSTOM_APPROX(finite_diff_deriv_l, input_coords_deriv_l,
  //                                finite_difference_approx);

  //   std::cout << "This is finite diff deriv_l:"
  //             << "\n"
  //             << finite_diff_deriv_l << "\n";
  //   //   std::cout << "This is input_coords_l:"
  //   //             << "\n"
  //   //             << input_coords_l << "\n";
  //   std::cout << "This is deriv_l: "
  //             << "\n"
  //             << std::setprecision(12) << deriv_l << "\n";

  // const std::array<double, 3> lower_bound{{0.82, 1.24, 1.32}};
  // const size_t grid_size = 8;
  // const std::array<double, 3> upper_bound{{0.8, 1.22, 1.30}};
  // TestHelpers::VerifyGrSolution::verify_time_independent_einstein_solution(
  //     solution, grid_size, lower_bound, upper_bound,
  //     std::numeric_limits<double>::epsilon() * 1.e5);
}
