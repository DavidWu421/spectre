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
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};

  // non perturbed spatial coordinates
  const auto x = spatial_coords<Frame::Inertial>(used_for_size);

  // Set up the solution, computer object, and cache object
  gr::Solutions::SphKerrSchild solution(mass, spin, center);

  // non perturbed computer
  gr::Solutions::SphKerrSchild::IntermediateComputer sks_computer(solution, x);
  gr::Solutions::SphKerrSchild::IntermediateVars<DataVector, Frame::Inertial>
      cache(used_for_sizet);

  // Functions outputs and tests

  // x_sph_minus_center test - non perturbed
  auto x_sph_minus_center = spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer(make_not_null(&x_sph_minus_center), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::x_sph_minus_center<
                   DataVector, Frame::Inertial>{});

  std::cout << "x_sph_minus_center: "
            << "\n"
            << x_sph_minus_center << "\n";

  // r_squared test - non perturbed
  Scalar<DataVector> r_squared(3_st, 0.);
  sks_computer(
      make_not_null(&r_squared), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::r_squared<DataVector>{});

  std::cout << "This is r_squared: "
            << "\n"
            << r_squared << "\n";

  // r test - non perturbed
  Scalar<DataVector> r(3_st, 0.);
  sks_computer(make_not_null(&r), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::r<DataVector>{});

  std::cout << "This is r: "
            << "\n"
            << r << "\n";

  // rho test - non perturbed
  Scalar<DataVector> rho(3_st, 0.);
  sks_computer(make_not_null(&rho), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::rho<DataVector>{});

  std::cout << "This is rho: "
            << "\n"
            << rho << "\n";

  // matrix_F test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_F{1_st, 0.};
  sks_computer(
      make_not_null(&matrix_F), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_F<DataVector,
                                                            Frame::Inertial>{});

  std::cout << "This is matrix F: "
            << "\n"
            << matrix_F << "\n";

  // matrix_P test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_P{1_st, 0.};
  sks_computer(
      make_not_null(&matrix_P), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_P<DataVector,
                                                            Frame::Inertial>{});
  std::cout << "This is matrix P: "
            << "\n"
            << matrix_P << "\n";

  // jacobian test - non perturbed
  tnsr::Ij<DataVector, 3, Frame::Inertial> jacobian{1_st, 0.};
  sks_computer(
      make_not_null(&jacobian), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::jacobian<DataVector,
                                                            Frame::Inertial>{});

  std::cout << "This is the jacobian: " << std::setprecision(16) << "\n"
            << jacobian << std::endl;

  // matrix_D test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_D{1_st, 0.};
  sks_computer(
      make_not_null(&matrix_D), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_D<DataVector,
                                                            Frame::Inertial>{});

  std::cout << "This is matrix D: "
            << "\n"
            << matrix_D << "\n";

  // matrix_C test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_C{1_st, 0.};
  sks_computer(
      make_not_null(&matrix_C), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_C<DataVector,
                                                            Frame::Inertial>{});

  std::cout << "This is matrix C: "
            << "\n"
            << matrix_C << "\n";

  // deriv_jacobian test
  tnsr::iJk<DataVector, 3, Frame::Inertial> deriv_jacobian{1_st, 0.};
  sks_computer(make_not_null(&deriv_jacobian), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::deriv_jacobian<
                   DataVector, Frame::Inertial>{});

  std::cout << "This is the deriv_jacobian: "
            << "\n"
            << deriv_jacobian << std::endl;

  // matrix_Q test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_Q{1_st, 0.};
  sks_computer(
      make_not_null(&matrix_Q), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_Q<DataVector,
                                                            Frame::Inertial>{});

  // matrix_G1 test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_G1{1_st, 0.};
  sks_computer(make_not_null(&matrix_G1), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::matrix_G1<
                   DataVector, Frame::Inertial>{});

  // a_dot_x test - non perturbed
  Scalar<DataVector> a_dot_x(3_st, 0.);
  sks_computer(
      make_not_null(&a_dot_x), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::a_dot_x<DataVector>{});

  std::cout << "This is a_dot_x: "
            << "\n"
            << a_dot_x << std::endl;

  // matrix_G2 test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_G2{1_st, 0.};
  sks_computer(make_not_null(&matrix_G2), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::matrix_G2<
                   DataVector, Frame::Inertial>{});

  std::cout << "This is matrix G2: "
            << "\n"
            << matrix_G2 << "\n";

  // G1_dot_x test
  tnsr::I<DataVector, 3, Frame::Inertial> G1_dot_x{3_st, 0.};
  sks_computer(
      make_not_null(&G1_dot_x), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::G1_dot_x<DataVector,
                                                            Frame::Inertial>{});

  std::cout << "This is matrix G2: "
            << "\n"
            << matrix_G2 << "\n";

  // G2_dot_x test
  tnsr::i<DataVector, 3, Frame::Inertial> G2_dot_x{3_st, 0.};
  sks_computer(
      make_not_null(&G2_dot_x), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::G2_dot_x<DataVector,
                                                            Frame::Inertial>{});

  std::cout << "This is G2 dot x: "
            << "\n"
            << G2_dot_x << "\n";

  // inv_jacobian test - non perturbed
  tnsr::Ij<DataVector, 3, Frame::Inertial> inv_jacobian{1_st, 0.};
  sks_computer(make_not_null(&inv_jacobian), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::inv_jacobian<
                   DataVector, Frame::Inertial>{});

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

  std::cout << "This is matrix E2: "
            << "\n"
            << matrix_E2 << "\n";

  //   deriv_inv_jacobian test - non perturbed
  tnsr::iJk<DataVector, 3, Frame::Inertial> deriv_inv_jacobian{1_st, 0.};
  sks_computer(make_not_null(&deriv_inv_jacobian), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::deriv_inv_jacobian<
                   DataVector, Frame::Inertial>{});

  std::cout << "This is deriv inv jacobian: "
            << "\n"
            << deriv_inv_jacobian << "\n";

  //  H test - non perturbed
  Scalar<DataVector> H{0.};
  sks_computer(make_not_null(&H), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::H<DataVector>{});

  std::cout << "This is scalar H: "
            << "\n"
            << H << "\n";

  // x_kerr_schild test - non perturbed
  auto x_kerr_schild = spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer(make_not_null(&x_kerr_schild), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::x_kerr_schild<
                   DataVector, Frame::Inertial>{});

  std::cout << "This is x_kerr_schild: "
            << "\n"
            << x_kerr_schild << "\n";

  // a_cross_x test - non perturbed
  auto a_cross_x = spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer(make_not_null(&a_cross_x), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::a_cross_x<
                   DataVector, Frame::Inertial>{});

  std::cout << "This is a_cross_x: "
            << "\n"
            << a_cross_x << "\n";

  // kerr_schild_l test - non perturbed
  auto kerr_schild_l = spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer(make_not_null(&kerr_schild_l), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::kerr_schild_l<
                   DataVector, Frame::Inertial>{});

  std::cout << "This is kerr_schild_l: "
            << "\n"
            << kerr_schild_l << "\n";

  // sph_kerr_schild)l_lower test - non perturbed
  tnsr::i<DataVector, 4, Frame::Inertial> sph_kerr_schild_l_lower{
      used_for_size};
  sks_computer(
      make_not_null(&sph_kerr_schild_l_lower), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::sph_kerr_schild_l_lower<
          DataVector, Frame::Inertial>{});

  std::cout << "This is sph_kerr_schild_l_lower: "
            << "\n"
            << sph_kerr_schild_l_lower << "\n";

  // sph_kerr_schild_l_upper test - non perturbed
  tnsr::I<DataVector, 4, Frame::Inertial> sph_kerr_schild_l_upper{
      used_for_size};
  sks_computer(
      make_not_null(&sph_kerr_schild_l_upper), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::sph_kerr_schild_l_upper<
          DataVector, Frame::Inertial>{});

  std::cout << "This is sph_kerr_schild_l_upper: "
            << "\n"
            << sph_kerr_schild_l_upper << "\n";

  // deriv_H test - non perturbed
  tnsr::I<DataVector, 4, Frame::Inertial> deriv_H{used_for_size};
  sks_computer(
      make_not_null(&deriv_H), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::deriv_H<DataVector,
                                                           Frame::Inertial>{});

  std::cout << "This is deriv_H: "
            << "\n"
            << deriv_H << "\n";

  //   deriv_l - non perturbed
  tnsr::ij<DataVector, 4, Frame::Inertial> deriv_l{used_for_size};
  sks_computer(
      make_not_null(&deriv_l), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::deriv_l<DataVector,
                                                           Frame::Inertial>{});

  std::cout << "This is deriv_l: "
            << "\n"
            << deriv_l << "\n";

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

  CHECK_ITERABLE_CUSTOM_APPROX(finite_diff_jacobian, input_coords_jacobian,
                               finite_difference_approx);

  std::cout << "finite diff jacobian: " << finite_diff_jacobian << "\n";

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

  CHECK_ITERABLE_CUSTOM_APPROX(finite_diff_deriv_jacobian,
                               input_coords_deriv_jacobian,
                               finite_difference_approx);

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

  CHECK_ITERABLE_CUSTOM_APPROX(finite_diff_deriv_H, input_coords_deriv_H,
                               finite_difference_approx);

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

  std::cout << "This is finite diff deriv_l:"
            << "\n"
            << finite_diff_deriv_l << "\n";
  //   std::cout << "This is input_coords_l:"
  //             << "\n"
  //             << input_coords_l << "\n";
  std::cout << "This is deriv_l: "
            << "\n"
            << deriv_l << "\n";

  // const std::array<double, 3> lower_bound{{0.82, 1.24, 1.32}};
  // const size_t grid_size = 8;
  // const std::array<double, 3> upper_bound{{0.8, 1.22, 1.30}};
  // TestHelpers::VerifyGrSolution::verify_time_independent_einstein_solution(
  //     solution, grid_size, lower_bound, upper_bound,
  //     std::numeric_limits<double>::epsilon() * 1.e5);

  // FINITE DIFFERENCE TEST

  // Tests f(X,Y,Z) = (X^1+Y^2+Z^3, 2X^2+3Y+Z, 3X^1+2Y^3+5Z^3) with X=1.1,
  // Y=2.2, Z=3.3
  double X = 1.1;
  double Y = 2.2;
  double Z = 3.3;
  double Xdx = 1.1 + .0001;
  double Ydx = 2.2 + .0001;
  double Zdx = 3.3 + .0001;

  auto input_vector =
      make_with_value<tnsr::I<double, 3, Frame::Inertial>>(1, 0.0);
  input_vector[0] = pow(X, 1) + pow(Y, 2) + pow(Z, 3);
  input_vector[1] = 2 * pow(X, 2) + 3 * pow(Y, 1) + pow(Z, 1);
  input_vector[2] = 3 * pow(X, 1) + 2 * pow(Y, 3) + 5 * pow(Z, 3);

  tnsr::Ij<DataVector, 3, Frame::Inertial> pert_matrix{1_st, 0.};
  pert_matrix.get(0, 0) = pow(Xdx, 1) + pow(Y, 2) + pow(Z, 3);
  pert_matrix.get(1, 0) = 2 * pow(Xdx, 2) + 3 * pow(Y, 1) + pow(Z, 1);
  pert_matrix.get(2, 0) = 3 * pow(Xdx, 1) + 2 * pow(Y, 3) + 5 * pow(Z, 3);
  pert_matrix.get(0, 1) = pow(X, 1) + pow(Ydx, 2) + pow(Z, 3);
  pert_matrix.get(1, 1) = 2 * pow(X, 2) + 3 * pow(Ydx, 1) + pow(Z, 1);
  pert_matrix.get(2, 1) = 3 * pow(X, 1) + 2 * pow(Ydx, 3) + 5 * pow(Z, 3);
  pert_matrix.get(0, 2) = pow(X, 1) + pow(Y, 2) + pow(Zdx, 3);
  pert_matrix.get(1, 2) = 2 * pow(X, 2) + 3 * pow(Y, 1) + pow(Zdx, 1);
  pert_matrix.get(2, 2) = 3 * pow(X, 1) + 2 * pow(Y, 3) + 5 * pow(Zdx, 3);

  auto perturbation_vector =
      make_with_value<tnsr::I<double, 3, Frame::Inertial>>(1, 0.0001);

  const auto test_jacobian =
      pypp::call<tnsr::Ij<DataVector, 3, Frame::Inertial>>(
          "General_Finite_Difference", "check_finite_difference_rank1",
          input_vector, pert_matrix, perturbation_vector);

  std::cout << "This is the test func jacobian: " << test_jacobian << "\n";

  tnsr::Ij<DataVector, 3, Frame::Inertial> real_jac{1_st, 0.};
  real_jac.get(0, 0) = 1;
  real_jac.get(1, 0) = 4 * X;
  real_jac.get(2, 0) = 3 * X;
  real_jac.get(0, 1) = 2 * Y;
  real_jac.get(1, 1) = 3;
  real_jac.get(2, 1) = 6 * pow(Y, 2);
  real_jac.get(0, 2) = 3 * pow(Z, 2);
  real_jac.get(1, 2) = 1;
  real_jac.get(2, 2) = 15 * pow(Z, 2);

  std::cout << "This is the real func jacobian: " << real_jac << "\n";

  // // Tests f(X,Y,Z) = (X^4+Y^5+Z^6, X^7+Y^8+Z^9, X^1+Y^2+Z^3) with
  // X=1.1,Y=2.2, Z=3.3
  //   double X = 1.1;
  //   double Y = 2.2;
  //   double Z = 3.3;
  //   double Xdx = 1.1+.0001;
  //   double Ydx = 2.2+.0001;
  //   double Zdx = 3.3+.0001;

  //   auto input_vector =
  //     make_with_value<tnsr::I<double, 3, Frame::Inertial>>(1, 0.0);
  //   input_vector[0] = pow(X,4)+pow(Y,5)+pow(Z,6);
  //   input_vector[1] = pow(X,7)+pow(Y,8)+pow(Z,9);
  //   input_vector[2] = pow(X,1)+pow(Y,2)+pow(Z,3);

  //   tnsr::Ij<DataVector, 3, Frame::Inertial> pert_matrix{1_st, 0.};
  //   pert_matrix.get(0,0) = pow(Xdx,4)+pow(Y,5)+pow(Z,6);
  //   pert_matrix.get(1,0) = pow(Xdx,7)+pow(Y,8)+pow(Z,9);
  //   pert_matrix.get(2,0) = pow(Xdx,1)+pow(Y,2)+pow(Z,3);
  //   pert_matrix.get(0,1) = pow(X,4)+pow(Ydx,5)+pow(Z,6);
  //   pert_matrix.get(1,1) = pow(X,7)+pow(Ydx,8)+pow(Z,9);
  //   pert_matrix.get(2,1) = pow(X,1)+pow(Ydx,2)+pow(Z,3);
  //   pert_matrix.get(0,2) = pow(X,4)+pow(Y,5)+pow(Zdx,6);
  //   pert_matrix.get(1,2) = pow(X,7)+pow(Y,8)+pow(Zdx,9);
  //   pert_matrix.get(2,2) = pow(X,1)+pow(Y,2)+pow(Zdx,3);

  //   auto perturbation_vector =
  //       make_with_value<tnsr::I<double, 3, Frame::Inertial>>(1, 0.0001);

  //   const auto test_jacobian =
  //       pypp::call<tnsr::Ij<DataVector, 3, Frame::Inertial>>(
  //           "General_Finite_Difference", "check_finite_difference_rank1",
  //           input_vector, pert_matrix, perturbation_vector);

  //   std::cout << "This is the test func jacobian: " << test_jacobian << "\n";

  //   tnsr::Ij<DataVector, 3, Frame::Inertial> real_jac{1_st, 0.};
  //   real_jac.get(0,0) = 4*pow(X,3);
  //   real_jac.get(1,0) = 7*pow(X,6);
  //   real_jac.get(2,0) = 1;
  //   real_jac.get(0,1) = 5*pow(Y,4);
  //   real_jac.get(1,1) = 8*pow(Y,7);
  //   real_jac.get(2,1) = 2*Y;
  //   real_jac.get(0,2) = 6*pow(Z,5);
  //   real_jac.get(1,2) = 9*pow(Z,8);
  //   real_jac.get(2,2) = 3*pow(Z,2);

  //   std::cout << "This is the real func jacobian: " << real_jac << "\n";
}
