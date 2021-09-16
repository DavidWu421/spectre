// Distributed under the MIT License.
// See LICENSE.txt for details.
#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>
#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/GeneralRelativity/VerifyGrSolution.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphKerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

#include "DataStructures/Tensor/EagerMath/CrossProduct.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Solutions.hpp"

#include <cmath>
#include <iostream>
#include <typeinfo>

namespace {

template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> spatial_coords(
    const DataType& used_for_size) noexcept {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  get<0>(x) = 1.1;
  get<1>(x) = 3.2;
  get<2>(x) = 5.3;
  return x;
}

template <typename Frame, typename DataType>
void test_sph_kerr_schild(const DataType& used_for_size) noexcept {
  // Parameters for SphKerrSchild solution
  const double mass = 1.01;
  const std::array<double, 3> spin{{0.0, 0.0, 0.0}};
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};
  const auto x = spatial_coords<Frame>(used_for_size);

  // Evaluate solution, instantiating a spherical kerrschild and calling
  // constructor
  gr::Solutions::SphKerrSchild solution(mass, spin, center);
}

// const tnsr::I<DataVector, 3, Frame::Inertial> make_x_coords() noexcept {
//   tnsr::I<DataVector, 3, Frame::Inertial> empty{3, 0.0};

//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = 0; j < 3; ++j) {
//       if (i == j) {
//         empty.get(i)[j] = 2.0;
//       }
//     }
//   }
//   return empty;
// }

// const DataVector r_squared(3, 4.0);

// const DataVector make_r() {
//   const DataVector r(3, sqrt(r_squared[0]));
//   return r;
// }

// const DataVector make_spin_a() {
//   DataVector spin_a(3, 0.0);
//   spin_a[0] = 1.;
//   spin_a[1] = 1.;
//   spin_a[2] = 1.0;
//   return spin_a;
// }

// const DataVector make_a_squared_array() {
//   auto spin_a = make_spin_a();
//   DataVector empty(3, 0.0);
//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = 0; j < 3; ++j) {
//       empty[i] += spin_a[j] * spin_a[j];
//     }
//   }
//   return empty;
// }

// auto make_a_squared() {
//   auto spin_a = make_spin_a();
//   auto a_squared =
//       std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
//   return a_squared;
// }

// const DataVector make_rho() {
//   const auto a_squared = make_a_squared_array();
//   DataVector empty(3, 0.0);
//   for (size_t i = 0; i < 3; ++i) {
//     empty[i] = sqrt(r_squared[i] + a_squared[i]);
//   }
//   return empty;
// }

// const DataVector make_a_dot_x() {
//   const auto spin_a = make_spin_a();
//   const auto x_coords = make_x_coords();
//   DataVector dot_product(3, 0.0);
//   for (size_t i = 0; i < 3; ++i) {
//     auto x_coords_vector = x_coords[i];
//     dot_product[i] = std::inner_product(spin_a.begin(), spin_a.end(),
//                                         x_coords_vector.begin(), 0.);
//   }
//   return dot_product;
// }

// const DataVector make_a_dot_x(){
//   const auto spin_a = make_spin_a();
//   const auto x_coords = make_x_coords();
//   DataVector inner_product(3, 0.0);
//   for (size_t i = 0; i < 3; ++i){
//     auto x_coords_vector = x_coords[i];
//     inner_product[i] = dot_product(spin_a, x_coords_vector);
//   }
//   return inner_product;
// }

// const tnsr::Ij<DataVector, 3, Frame::Inertial> make_matrix_F() {
//   tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_F{9, 0.0};
//   // Setup
//   auto r = make_r();
//   auto rho = make_rho();
//   auto spin_a = make_spin_a();
//   auto a_squared = make_a_squared_array();
//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = 0; j < 3; ++j) {
//       matrix_F.get(i, j) = -1. / rho / cube(r);
//     }
//   }
//   // F Matrix
//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = 0; j < 3; ++j) {
//       if (i == j) {
//         matrix_F.get(i, j) *= (a_squared[i] - spin_a[i] * spin_a[j]);
//       } else {
//         matrix_F.get(i, j) *= -spin_a[i] * spin_a[j];
//       }
//     }
//   }
//   return matrix_F;
// }

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.SphKerrSchild",
                  "[PointwiseFunctions][Unit]") {
  // Evaluate Solution
  const DataVector used_for_size(3);
  const double used_for_size_double = used_for_size.size();
  const double mass = 1.01;
  const std::array<double, 3> spin{{0.2, 0.3, 0.4}};
  const std::array<double, 3> center{{0.1, 1.2, 2.3}};
  const auto x = spatial_coords<Frame::Inertial>(used_for_size);
  const double null_vector_0 = -1.0;
  // const double t = 1.0;

  // Set up the solution, computer objet, and cache object
  gr::Solutions::SphKerrSchild solution(mass, spin, center);
  gr::Solutions::SphKerrSchild::IntermediateComputer sks_computer(
      solution, x, null_vector_0);
  gr::Solutions::SphKerrSchild::IntermediateVars<DataVector, Frame::Inertial>
      cache(solution, x);

  // x_sph_minus_center test
  auto x_sph_minus_center = spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer(make_not_null(&x_sph_minus_center), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::x_sph_minus_center<
                   DataVector, Frame::Inertial>{});

  // r_squared test
  Scalar<DataVector> r_squared(3, 0.);
  sks_computer(
      make_not_null(&r_squared), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::r_squared<DataVector>{});

  // r test
  Scalar<DataVector> r(3, 0.);
  sks_computer(make_not_null(&r), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::r<DataVector>{});

  // rho test
  Scalar<DataVector> rho(3, 0.);
  sks_computer(make_not_null(&rho), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::rho<DataVector>{});

  // a_dot_x test
  Scalar<DataVector> a_dot_x(3, 0.);
  sks_computer(
      make_not_null(&a_dot_x), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::a_dot_x<DataVector>{});

  // matrix_F test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_F{1, 0.};
  sks_computer(
      make_not_null(&matrix_F), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_F<DataVector,
                                                            Frame::Inertial>{});

  // matrix_P test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_P{1, 0.};
  sks_computer(
      make_not_null(&matrix_P), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_P<DataVector,
                                                            Frame::Inertial>{});

  // jacobian test
  tnsr::Ij<DataVector, 3, Frame::Inertial> jacobian{1, 0.};
  sks_computer(
      make_not_null(&jacobian), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::jacobian<DataVector,
                                                            Frame::Inertial>{});

  // matrix_D test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_D{1, 0.};
  sks_computer(
      make_not_null(&matrix_D), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_D<DataVector,
                                                            Frame::Inertial>{});

  // matrix_C test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_C{1, 0.};
  sks_computer(
      make_not_null(&matrix_C), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_C<DataVector,
                                                            Frame::Inertial>{});

  // deriv_jacobian test: Needs to be checked
  tnsr::ijK<DataVector, 3, Frame::Inertial> deriv_jacobian{1, 0.};
  sks_computer(make_not_null(&deriv_jacobian), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::deriv_jacobian<
                   DataVector, Frame::Inertial>{});

  // matrix_Q test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_Q{1, 0.};
  sks_computer(
      make_not_null(&matrix_Q), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_Q<DataVector,
                                                            Frame::Inertial>{});

  // matrix_G1 test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_G1{1, 0.};
  sks_computer(make_not_null(&matrix_G1), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::matrix_G1<
                   DataVector, Frame::Inertial>{});

  // s_number test
  Scalar<DataVector> s_number{1, 0.};
  sks_computer(
      make_not_null(&s_number), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::internal_tags::s_number<
          DataVector>{});

  // matrix_G2 test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_G2{1, 0.};
  sks_computer(make_not_null(&matrix_G2), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::matrix_G2<
                   DataVector, Frame::Inertial>{});

  // G1_dot_x test
  tnsr::I<DataVector, 3, Frame::Inertial> G1_dot_x{3, 0.};
  sks_computer(
      make_not_null(&G1_dot_x), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::G1_dot_x<DataVector,
                                                            Frame::Inertial>{});

  // G2_dot_x test
  tnsr::i<DataVector, 3, Frame::Inertial> G2_dot_x{3, 0.};
  sks_computer(
      make_not_null(&G2_dot_x), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::G2_dot_x<DataVector,
                                                            Frame::Inertial>{});

  // inv_jacobian test
  tnsr::Ij<DataVector, 3, Frame::Inertial> inv_jacobian{1, 0.};
  sks_computer(make_not_null(&inv_jacobian), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::inv_jacobian<
                   DataVector, Frame::Inertial>{});

  // matrix_E1 test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_E1{1, 0.};
  sks_computer(make_not_null(&matrix_E1), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::matrix_E1<
                   DataVector, Frame::Inertial>{});

  // matrix_E2 test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_E2{1, 0.};
  sks_computer(make_not_null(&matrix_E2), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::matrix_E2<
                   DataVector, Frame::Inertial>{});

  // x_kerr_schild test
  auto x_kerr_schild = spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer(make_not_null(&x_kerr_schild), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::x_kerr_schild<
                   DataVector, Frame::Inertial>{});

  // deriv_inv_jacobian test
  tnsr::ijK<DataVector, 3, Frame::Inertial> deriv_inv_jacobian{1, 0.};
  sks_computer(make_not_null(&deriv_inv_jacobian), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::deriv_inv_jacobian<
                   DataVector, Frame::Inertial>{});

  auto a_cross_x = spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer(make_not_null(&a_cross_x), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::a_cross_x<
                   DataVector, Frame::Inertial>{});

  auto kerr_schild_l = spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer(make_not_null(&kerr_schild_l), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::kerr_schild_l<
                   DataVector, Frame::Inertial>{});

  tnsr::I<DataVector, 4, Frame::Inertial> sph_kerr_schild_l_upper{
      used_for_size_double, 0.};
  sks_computer(
      make_not_null(&sph_kerr_schild_l_upper), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::sph_kerr_schild_l_upper<
          DataVector, Frame::Inertial>{});

  tnsr::i<DataVector, 4, Frame::Inertial> sph_kerr_schild_l_lower{
      used_for_size_double, 0.};
  sks_computer(
      make_not_null(&sph_kerr_schild_l_lower), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::sph_kerr_schild_l_lower<
          DataVector, Frame::Inertial>{});

  Scalar<DataVector> H{1, 0.};
  sks_computer(make_not_null(&H), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::internal_tags::H<
                   DataVector>{});

  tnsr::I<DataVector, 4, Frame::Inertial> deriv_H{used_for_size_double, 0.};
  sks_computer(
      make_not_null(&deriv_H), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::deriv_H<DataVector,
                                                           Frame::Inertial>{});

  tnsr::Ij<DataVector, 4, Frame::Inertial> deriv_l{used_for_size_double, 0.};
  sks_computer(
      make_not_null(&deriv_l), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::deriv_l<DataVector,
                                                           Frame::Inertial>{});

  Scalar<DataVector> lapse_squared{1, 0.};
  sks_computer(
      make_not_null(&lapse_squared), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::internal_tags::lapse_squared<
          DataVector>{});

  Scalar<DataVector> lapse{1, 0.};
  sks_computer(make_not_null(&lapse), make_not_null(&cache),
               gr::Tags::Lapse<DataVector>{});

  Scalar<DataVector> deriv_lapse_multiplier{1, 0.};
  sks_computer(make_not_null(&deriv_lapse_multiplier), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::internal_tags::
                   deriv_lapse_multiplier<DataVector>{});

  Scalar<DataVector> shift_multiplier{1, 0.};
  sks_computer(make_not_null(&shift_multiplier), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::internal_tags::
                   shift_multiplier<DataVector>{});

  tnsr::I<DataVector, 3, Frame::Inertial> shift{used_for_size_double, 0.};
  sks_computer(make_not_null(&shift), make_not_null(&cache),
               gr::Tags::Shift<3, Frame::Inertial, DataVector>{});

  tnsr::iJ<DataVector, 3, Frame::Inertial> deriv_shift{used_for_size_double,
                                                       0.};
  sks_computer(
      make_not_null(&deriv_shift), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::DerivShift<DataVector, Frame::Inertial>{});

  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{used_for_size_double,
                                                          0.};
  sks_computer(make_not_null(&spatial_metric), make_not_null(&cache),
               gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>{});

  tnsr::ijj<DataVector, 3, Frame::Inertial> deriv_spatial_metric{
      used_for_size_double, 0.};
  sks_computer(
      make_not_null(&deriv_spatial_metric), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::DerivSpatialMetric<DataVector,
                                                       Frame::Inertial>{});

  tnsr::ii<DataVector, 3, Frame::Inertial> dt_spatial_metric{
      used_for_size_double, 0.};
  sks_computer(
      make_not_null(&dt_spatial_metric), make_not_null(&cache),
      ::Tags::dt<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>{});

  //   const std::array<double, 3> lower_bound{{0.82, 1.24, 1.32}};
  //   const size_t grid_size = 12;
  //   const std::array<double, 3> upper_bound{{0.8, 1.22, 1.30}};
  //   TestHelpers::VerifyGrSolution::verify_time_independent_einstein_solution(
  //       solution, grid_size, lower_bound, upper_bound,
  //       std::numeric_limits<double>::epsilon() * 1.e5);
}
