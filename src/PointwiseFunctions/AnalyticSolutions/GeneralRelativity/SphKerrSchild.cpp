// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphKerrSchild.hpp"

#include <cmath>  // IWYU pragma: keep
#include <iostream>
#include <numeric>
#include <ostream>
#include <typeinfo>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/CrossProduct.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/StdHelpers.hpp"

#include "DataStructures/Tensor/Tensor.hpp"

#include "Utilities/ContainerHelpers.hpp"

#include <iostream>

namespace gr::Solutions {

SphKerrSchild::SphKerrSchild(const double mass,
                             SphKerrSchild::Spin::type dimensionless_spin,
                             SphKerrSchild::Center::type center,
                             const Options::Context& context)
    : mass_(mass),
      // clang-tidy: do not std::move trivial types.
      dimensionless_spin_(std::move(dimensionless_spin)),  // NOLINT
      // clang-tidy: do not std::move trivial types.
      center_(std::move(center))  // NOLINT
{
  const double spin_magnitude = magnitude(dimensionless_spin_);
  if (spin_magnitude > 1.0) {
    PARSE_ERROR(context, "Spin magnitude must be < 1. Given spin: "
                             << dimensionless_spin_ << " with magnitude "
                             << spin_magnitude);
  }
  if (mass_ < 0.0) {
    PARSE_ERROR(context, "Mass must be non-negative. Given mass: " << mass_);
  }
}

void SphKerrSchild::pup(PUP::er& p) noexcept {
  p | mass_;
  p | dimensionless_spin_;
  p | center_;
}

template <typename DataType, typename Frame>
SphKerrSchild::IntermediateComputer<DataType, Frame>::IntermediateComputer(
    const SphKerrSchild& solution, const tnsr::I<DataType, 3, Frame>& x,
    const double null_vector_0) noexcept
    : solution_(solution), x_(x), null_vector_0_(null_vector_0) {}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_sph_minus_center,
    const gsl::not_null<CachedBuffer*> /*cache*/,
    internal_tags::x_sph_minus_center<DataType, Frame> /*meta*/)
    const noexcept {
  *x_sph_minus_center = x_;
  for (size_t d = 0; d < 3; ++d) {
    x_sph_minus_center->get(d) -= gsl::at(solution_.center(), d);
  }

  std::cout << "this is x_sph_minus_center " << *x_sph_minus_center << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> r_squared,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::r_squared<DataType> /*meta*/) const noexcept {
  const auto& x_sph_minus_center =
      cache->get_var(internal_tags::x_sph_minus_center<DataType, Frame>{});

  r_squared->get() = square(x_sph_minus_center.get(0));
  for (size_t i = 1; i < 3; ++i) {
    r_squared->get() += square(x_sph_minus_center.get(i));
  }
  std::cout << "this is r_squared " << *r_squared << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::r<DataType> /*meta*/) const noexcept {
  const auto& r_squared =
      get(cache->get_var(internal_tags::r_squared<DataType>{}));

  get(*r) = sqrt(r_squared);

  std::cout << "this is r " << *r << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> rho,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::rho<DataType> /*meta*/) const noexcept {
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto& r_squared =
      get(cache->get_var(internal_tags::r_squared<DataType>{}));
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
  get(*rho) = sqrt(r_squared + a_squared);

  std::cout << "this is rho " << *rho << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_F,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_F<DataType, Frame> /*meta*/) const noexcept {
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
  const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      matrix_F->get(i, j) = -1. / rho / cube(r);
    }
  }
  // F matrix
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (i == j) {
        matrix_F->get(i, j) *= (a_squared - spin_a[i] * spin_a[j]);
      } else {
        matrix_F->get(i, j) *= -spin_a[i] * spin_a[j];
      }
    }
  }
  std::cout << "this is matrix_F:"
            << "\n"
            << *matrix_F << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_P,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_P<DataType, Frame> /*meta*/) const noexcept {
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      matrix_P->get(i, j) = -1. / (rho + r) / r;
    }
  }
  // P matrix
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (i == j) {
        matrix_P->get(i, j) *= spin_a[i] * spin_a[j];
        matrix_P->get(i, j) += rho / r;
      } else {
        matrix_P->get(i, j) *= spin_a[i] * spin_a[j];
      }
    }
  }
  std::cout << "this is matrix_P:"
            << "\n"
            << *matrix_P << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> jacobian,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::jacobian<DataType, Frame> /*meta*/) const noexcept {
  const auto& x_sph_minus_center =
      cache->get_var(internal_tags::x_sph_minus_center<DataType, Frame>{});
  const auto& matrix_P =
      cache->get_var(internal_tags::matrix_P<DataType, Frame>{});
  const auto& matrix_F =
      cache->get_var(internal_tags::matrix_F<DataType, Frame>{});

  // Jacobian
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      jacobian->get(i, j) = matrix_P.get(i, j);
      for (size_t k = 0; k < 3; ++k) {
        jacobian->get(i, j) += matrix_F.get(i, k) * x_sph_minus_center.get(k) *
                               x_sph_minus_center.get(j);
      }
    }
  }
  std::cout << "this is the jacobian:"
            << "\n"
            << *jacobian << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_D,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_D<DataType, Frame> /*meta*/) const noexcept {
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
  const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      matrix_D->get(i, j) = 1. / cube(rho) / r;
    }
  }
  // D matrix
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (i == j) {
        matrix_D->get(i, j) *= (a_squared - spin_a[i] * spin_a[j]);
      } else {
        matrix_D->get(i, j) *= -spin_a[i] * spin_a[j];
      }
    }
  }
  std::cout << "this is matrix_D:"
            << "\n"
            << *matrix_D << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_C,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_C<DataType, Frame> /*meta*/) const noexcept {
  const auto& matrix_F =
      cache->get_var(internal_tags::matrix_F<DataType, Frame>{});
  const auto& matrix_D =
      cache->get_var(internal_tags::matrix_D<DataType, Frame>{});

  // C matrix
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      matrix_C->get(i, j) = matrix_D.get(i, j) - 3. * matrix_F.get(i, j);
    }
  }
  std::cout << "this is matrix_C:"
            << "\n"
            << *matrix_C << "\n";
}

// TEST
template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ijK<DataType, 3, Frame>*> deriv_jacobian,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_jacobian<DataType, Frame> /*meta*/) const noexcept {
  const auto& matrix_C =
      cache->get_var(internal_tags::matrix_C<DataType, Frame>{});
  const auto& matrix_F =
      cache->get_var(internal_tags::matrix_F<DataType, Frame>{});
  const auto& x_sph_minus_center =
      cache->get_var(internal_tags::x_sph_minus_center<DataType, Frame>{});
  const auto& r_squared =
      get(cache->get_var(internal_tags::r_squared<DataType>{}));

  // deriv_Jacobian
  for (size_t k = 0; k < 3; ++k) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        deriv_jacobian->get(i, j, k) =
            matrix_F.get(i, j) * x_sph_minus_center.get(k) +
            matrix_F.get(i, k) * x_sph_minus_center.get(j);
        for (size_t m = 0; m < 3; ++m) {
          if (j == k) {  // j==k acts as a Kronecker delta
            deriv_jacobian->get(i, j, k) +=
                matrix_F.get(i, m) * x_sph_minus_center.get(m);
          }
          deriv_jacobian->get(i, j, k) +=
              matrix_C.get(i, m) * x_sph_minus_center.get(k) *
              x_sph_minus_center.get(m) * x_sph_minus_center.get(j) / r_squared;
        }
      }
    }
  }
  std::cout << "this is deriv_jacobian:"
            << "\n"
            << *deriv_jacobian << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_Q,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_Q<DataType, Frame> /*meta*/) const noexcept {
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));

  // Q matrix
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      matrix_Q->get(i, j) = 1. / (rho + r) / rho;
      if (i == j) {
        matrix_Q->get(i, j) *= spin_a[i] * spin_a[j];
        matrix_Q->get(i, j) += r / rho;
      } else {
        matrix_Q->get(i, j) *= spin_a[i] * spin_a[j];
      }
    }
  }
  std::cout << "this is matrix_Q:"
            << "\n"
            << *matrix_Q << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> a_dot_x,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::a_dot_x<DataType> /*meta*/) const noexcept {
  const auto& x_sph_minus_center =
      cache->get_var(internal_tags::x_sph_minus_center<DataType, Frame>{});

  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  get(*a_dot_x) = spin_a[0] * get<0>(x_sph_minus_center) +
                  spin_a[1] * get<1>(x_sph_minus_center) +
                  spin_a[2] * get<2>(x_sph_minus_center);

  std::cout << "this is spin_a " << spin_a << "\n";
  std::cout << "this is a_dot_x " << *a_dot_x << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_G1,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_G1<DataType, Frame> /*meta*/) const noexcept {
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
  const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      matrix_G1->get(i, j) = 1. / square(rho) / r;
      if (i == j) {
        matrix_G1->get(i, j) *= (a_squared - spin_a[i] * spin_a[j]);
      } else {
        matrix_G1->get(i, j) *= -spin_a[i] * spin_a[j];
      }
    }
  }
  std::cout << "this is matrix_G1:"
            << "\n"
            << *matrix_G1 << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> s_number,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::s_number<DataType> /*meta*/) const noexcept {
  const auto& r_squared =
      get(cache->get_var(internal_tags::r_squared<DataType>{}));
  const auto& a_dot_x = get(cache->get_var(internal_tags::a_dot_x<DataType>{}));

  get(*s_number) = r_squared + square(a_dot_x) / r_squared;

  std::cout << "this is s_number:"
            << "\n"
            << *s_number << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_G2,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_G2<DataType, Frame> /*meta*/) const noexcept {
  const auto& matrix_Q =
      cache->get_var(internal_tags::matrix_Q<DataType, Frame>{});
  const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));
  const auto& s_number =
      get(cache->get_var(internal_tags::s_number<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      matrix_G2->get(i, j) = (square(rho) / r) * matrix_Q.get(i, j) / s_number;
    }
  }
  std::cout << "this is matrix_G2:"
            << "\n"
            << *matrix_G2 << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> G1_dot_x,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::G1_dot_x<DataType, Frame> /*meta*/) const noexcept {
  const auto& x_sph_minus_center =
      cache->get_var(internal_tags::x_sph_minus_center<DataType, Frame>{});
  const auto& matrix_G1 =
      cache->get_var(internal_tags::matrix_G1<DataType, Frame>{});

  for (size_t i = 0; i < 3; ++i) {
    G1_dot_x->get(i) = matrix_G1.get(i, 0) * x_sph_minus_center.get(0);
    for (size_t m = 1; m < 3; ++m) {
      G1_dot_x->get(i) += matrix_G1.get(i, m) * x_sph_minus_center.get(m);
    }
  }
  std::cout << "this is G1_dot_x:"
            << "\n"
            << *G1_dot_x << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3, Frame>*> G2_dot_x,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::G2_dot_x<DataType, Frame> /*meta*/) const noexcept {
  const auto& x_sph_minus_center =
      cache->get_var(internal_tags::x_sph_minus_center<DataType, Frame>{});
  const auto& matrix_G2 =
      cache->get_var(internal_tags::matrix_G2<DataType, Frame>{});

  for (size_t i = 0; i < 3; ++i) {
    G2_dot_x->get(i) = matrix_G2.get(0, i) * x_sph_minus_center.get(0);
    for (size_t m = 1; m < 3; ++m) {
      G2_dot_x->get(i) += matrix_G2.get(m, i) * x_sph_minus_center.get(m);
    }
  }
  std::cout << "this is G2_dot_x:"
            << "\n"
            << *G2_dot_x << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> inv_jacobian,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::inv_jacobian<DataType, Frame> /*meta*/) const noexcept {
  const auto& matrix_Q =
      cache->get_var(internal_tags::matrix_Q<DataType, Frame>{});
  const auto& G1_dot_x =
      cache->get_var(internal_tags::G1_dot_x<DataType, Frame>{});
  const auto& G2_dot_x =
      cache->get_var(internal_tags::G2_dot_x<DataType, Frame>{});

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      inv_jacobian->get(i, j) =
          matrix_Q.get(i, j) + G1_dot_x.get(i) * G2_dot_x.get(j);
    }
  }
  std::cout << "this is inv_jacobian:"
            << "\n"
            << *inv_jacobian << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_E1,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_E1<DataType, Frame> /*meta*/) const noexcept {
  const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
  const auto& r_squared =
      get(cache->get_var(internal_tags::r_squared<DataType>{}));
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      matrix_E1->get(i, j) =
          -1. / square(rho) * (1. / r_squared + 2 / square(rho));
      if (i == j) {
        matrix_E1->get(i, j) *= (a_squared - spin_a[i] * spin_a[j]);
      } else {
        matrix_E1->get(i, j) *= -spin_a[i] * spin_a[j];
      }
    }
  }
  std::cout << "this is matrix_E1:"
            << "\n"
            << *matrix_E1 << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_E2,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_E2<DataType, Frame> /*meta*/) const noexcept {
  const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
  const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));
  const auto& a_dot_x = get(cache->get_var(internal_tags::a_dot_x<DataType>{}));
  const auto& s_number =
      get(cache->get_var(internal_tags::s_number<DataType>{}));
  const auto& matrix_G2 =
      cache->get_var(internal_tags::matrix_G2<DataType, Frame>{});
  const auto& matrix_P =
      cache->get_var(internal_tags::matrix_P<DataType, Frame>{});

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      matrix_E2->get(i, j) = ((-a_squared / square(rho) / r) -
                              2. / s_number * (r - square(a_dot_x) / cube(r))) *
                                 matrix_G2.get(i, j) +
                             1. / s_number * matrix_P.get(i, j);
    }
  }

  std::cout << "this is matrix_E2:"
            << "\n"
            << *matrix_E2 << "\n";
}

// deriv_inv_jacobian
template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ijK<DataType, 3, Frame>*> deriv_inv_jacobian,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_inv_jacobian<DataType, Frame> /*meta*/)
    const noexcept {
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto& matrix_D =
      cache->get_var(internal_tags::matrix_D<DataType, Frame>{});
  const auto& matrix_G1 =
      cache->get_var(internal_tags::matrix_G1<DataType, Frame>{});
  const auto& matrix_G2 =
      cache->get_var(internal_tags::matrix_G2<DataType, Frame>{});
  const auto& matrix_E1 =
      cache->get_var(internal_tags::matrix_E1<DataType, Frame>{});
  const auto& matrix_E2 =
      cache->get_var(internal_tags::matrix_E2<DataType, Frame>{});
  const auto& x_sph_minus_center =
      cache->get_var(internal_tags::x_sph_minus_center<DataType, Frame>{});
  const auto& a_dot_x = get(cache->get_var(internal_tags::a_dot_x<DataType>{}));
  const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));
  const auto& G1_dot_x =
      cache->get_var(internal_tags::G1_dot_x<DataType, Frame>{});
  const auto& G2_dot_x =
      cache->get_var(internal_tags::G2_dot_x<DataType, Frame>{});
  const auto& s_number =
      get(cache->get_var(internal_tags::s_number<DataType>{}));

  for (int k = 0; k < 3; ++k) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        deriv_inv_jacobian->get(i, j, k) =
            matrix_D.get(i, j) * x_sph_minus_center.get(k) +
            matrix_G1.get(i, k) * G2_dot_x.get(j) +
            matrix_G2.get(k, j) * G1_dot_x.get(i) -
            2. * a_dot_x * spin_a[k] / s_number / square(r) * G1_dot_x.get(i) *
                G2_dot_x.get(j);
        for (int m = 0; m < 3; ++m) {
          deriv_inv_jacobian->get(i, j, k) +=
              matrix_E1.get(i, m) * x_sph_minus_center.get(m) *
                  G2_dot_x.get(j) * x_sph_minus_center.get(k) / r +
              G1_dot_x.get(i) * x_sph_minus_center.get(k) *
                  x_sph_minus_center.get(m) * matrix_E2.get(m, j) / r;
        }
      }
    }
  }
  std::cout << "this is deriv_inv_jacobian:"
            << "\n"
            << *deriv_inv_jacobian << "\n";
}

// // TEST
template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_kerr_schild,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::x_kerr_schild<DataType, Frame> /*meta*/) const noexcept {
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto& x_sph_minus_center =
      cache->get_var(internal_tags::x_sph_minus_center<DataType, Frame>{});
  const auto& a_dot_x = get(cache->get_var(internal_tags::a_dot_x<DataType>{}));
  const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));
  const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    x_kerr_schild->get(i) = rho / r * x_sph_minus_center.get(i) -
                            spin_a[i] * a_dot_x / r / (rho + r);
  }

  std::cout << "this is x_kerr_schild:"
            << "\n"
            << *x_kerr_schild << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> a_cross_x,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::a_cross_x<DataType, Frame> /*meta*/) const noexcept {
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const tnsr::I<DataType, 3, Frame>& x_kerr_schild =
      cache->get_var(internal_tags::x_kerr_schild<DataType, Frame>{});

  tnsr::i<DataVector, 3, Frame> cross_tensor{1, 0.};
  tnsr::i<DataVector, 3, Frame> spin_tensor{1, 0.};

  for (size_t m = 0; m < get_size(get_element(x_kerr_schild, 0)); ++m) {
    for (size_t s = 0; s < 3; ++s) {
      spin_tensor[s] = spin_a[s];
      cross_tensor[s] = get_element(x_kerr_schild.get(s), m);
    }
    auto temp_cross_product = cross_product(spin_tensor, cross_tensor);
    for (size_t i = 0; i < 3; ++i) {
      get_element(a_cross_x->get(i), m) = get_element(temp_cross_product[i], 0);
    }
  }

  std::cout << "this is a_cross_x: " << *a_cross_x << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> kerr_schild_l,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::kerr_schild_l<DataType, Frame> /*meta*/) const noexcept {
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto& a_dot_x = get(cache->get_var(internal_tags::a_dot_x<DataType>{}));
  const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));
  const auto& a_cross_x =
      cache->get_var(internal_tags::a_cross_x<DataType, Frame>{});
  const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
  const auto& x_kerr_schild =
      cache->get_var(internal_tags::x_kerr_schild<DataType, Frame>{});

  for (size_t s = 0; s < get_size(get_element(x_kerr_schild, 0)); ++s) {
    const double den = 1. / square(get_element(rho, s));
    const double rboyer = get_element(r, s);

    for (int i = 0; i < 3; ++i) {
      get_element(kerr_schild_l->get(i), s) =
          den * (rboyer * get_element(x_kerr_schild.get(i), s) +
                 get_element(a_dot_x, s) * spin_a[i] / rboyer -
                 get_element(a_cross_x.get(i), s));
    }
  }
  std::cout << "this is kerr_schild_l: " << *kerr_schild_l << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 4, Frame>*> sph_kerr_schild_l_upper,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::sph_kerr_schild_l_upper<DataType, Frame> /*meta*/)
    const noexcept {
  const auto& kerr_schild_l =
      cache->get_var(internal_tags::kerr_schild_l<DataType, Frame>{});
  const auto& inv_jacobian =
      cache->get_var(internal_tags::inv_jacobian<DataType, Frame>{});

  sph_kerr_schild_l_upper->get(0) = -1.;
  for (size_t s = 0; s < get_size(get_element(kerr_schild_l, 0)); ++s) {
    for (size_t j = 0; j < 3; ++j) {
      get_element(sph_kerr_schild_l_upper->get(j + 1), s) = 0.;
      for (size_t i = 0; i < 3; ++i) {
        get_element(sph_kerr_schild_l_upper->get(j + 1), s) +=
            get_element(inv_jacobian.get(j, i), s) *
            get_element(kerr_schild_l.get(i), s);
      }
    }
  }

  std::cout << "this is sph kerr schild l upper" << *sph_kerr_schild_l_upper
            << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::i<DataType, 4, Frame>*> sph_kerr_schild_l_lower,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::sph_kerr_schild_l_lower<DataType, Frame> /*meta*/)
    const noexcept {
  const auto& kerr_schild_l =
      cache->get_var(internal_tags::kerr_schild_l<DataType, Frame>{});
  const auto& jacobian =
      cache->get_var(internal_tags::jacobian<DataType, Frame>{});

  sph_kerr_schild_l_lower->get(0) = 1.;
  for (size_t s = 0; s < get_size(get_element(kerr_schild_l, 0)); ++s) {
    for (size_t j = 0; j < 3; ++j) {
      get_element(sph_kerr_schild_l_lower->get(j + 1), s) = 0.;
      for (size_t i = 0; i < 3; ++i) {
        get_element(sph_kerr_schild_l_lower->get(j + 1), s) +=
            get_element(jacobian.get(i, j), s) *
            get_element(kerr_schild_l.get(i), s);
      }
    }
  }

  std::cout << "this is sph kerr schild l lower" << *sph_kerr_schild_l_lower
            << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> H,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::H<DataType> /*meta*/) const noexcept {
  const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));
  const auto& a_dot_x = get(cache->get_var(internal_tags::a_dot_x<DataType>{}));

  H->get() = solution_.mass() * cube(r) / pow(r, 4) + square(a_dot_x);
  std::cout << "this is H: " << *H << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 4, Frame>*> deriv_H,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_H<DataType, Frame> /*meta*/) const noexcept {
  const auto& x_kerr_schild =
      cache->get_var(internal_tags::x_kerr_schild<DataType, Frame>{});
  const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));
  const auto& a_dot_x = get(cache->get_var(internal_tags::a_dot_x<DataType>{}));
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto& H = cache->get_var(internal_tags::H<DataType>{});
  const auto& jacobian =
      cache->get_var(internal_tags::jacobian<DataType, Frame>{});

  deriv_H->get(0) = 0.;

  for (size_t s = 0; s < get_size(get_element(x_kerr_schild, 0)); ++s) {
    const double rboyer = get_element(r, s);
    const double drden =
        get_element(H[0], s) /
        solution_.mass();  // H has M as a factor, but dr does not.

    DataVector dr(3, 0.);
    for (size_t i = 0; i < 3; ++i) {
      dr[i] = drden * (get_element(x_kerr_schild.get(i), s) +
                       get_element(a_dot_x, s) * spin_a[i] / square(rboyer));
    }

    const double Hden = 1. / (pow(rboyer, 4) + square(get_element(a_dot_x, s)));
    const double fac = 3. / rboyer - 4. * cube(rboyer) * Hden;
    for (size_t i = 0; i < 3; ++i) {
      get_element(deriv_H->get(i + 1), s) =
          get_element(H[0], s) *
          (fac * dr[i] - 2. * Hden * get_element(a_dot_x, s) *
                             spin_a[i]);  // deriv_H in original KS
    }

    const double deriv_H_x = get_element(deriv_H->get(1), s);
    const double deriv_H_y = get_element(deriv_H->get(2), s);
    const double deriv_H_z = get_element(deriv_H->get(3), s);

    for (size_t j = 0; j < 3; ++j) {
      get_element(deriv_H->get(j + 1), s) =
          get_element(jacobian.get(0, j), s) * deriv_H_x +
          get_element(jacobian.get(1, j), s) * deriv_H_y +
          get_element(jacobian.get(2, j), s) * deriv_H_z;
    }  // deriv_H in Spherical KS
  }
  std::cout << "this is deriv_H: " << *deriv_H << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 4, Frame>*> deriv_l,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_l<DataType, Frame> /*meta*/) const noexcept {
  const auto& x_kerr_schild =
      cache->get_var(internal_tags::x_kerr_schild<DataType, Frame>{});
  const auto& r = get(cache->get_var(internal_tags::r<DataType>{}));
  const auto& a_dot_x = get(cache->get_var(internal_tags::a_dot_x<DataType>{}));
  const auto& rho = get(cache->get_var(internal_tags::rho<DataType>{}));
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto& kerr_schild_l =
      cache->get_var(internal_tags::kerr_schild_l<DataType, Frame>{});
  const auto& H = cache->get_var(internal_tags::H<DataType>{});
  const auto& jacobian =
      cache->get_var(internal_tags::jacobian<DataType, Frame>{});
  const auto& deriv_jacobian =
      cache->get_var(internal_tags::deriv_jacobian<DataType, Frame>{});

  for (size_t i = 0; i < 4; ++i) {
    deriv_l->get(i, 0) = 0.;
    deriv_l->get(0, i) = 0.;
  }

  tnsr::ij<DataVector, 3, Frame> temp_deriv_l{
      get_size(get_element(x_kerr_schild, 0)), 0.};

  for (size_t s = 0; s < get_size(get_element(x_kerr_schild, 0)); ++s) {
    const double den = 1. / square(get_element(rho, s));
    const double rboyer = get_element(r, s);
    const double drden = get_element(H[0], s) / solution_.mass();
    DataVector dr(3, 0.);
    for (size_t i = 0; i < 3; ++i) {
      dr[i] = drden * (get_element(x_kerr_schild.get(i), s) +
                       get_element(a_dot_x, s) * spin_a[i] / square(rboyer));
    }

    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        get_element(deriv_l->get(i + 1, j + 1), s) =
            den * ((get_element(x_kerr_schild.get(i), s) -
                    2. * rboyer * get_element(kerr_schild_l.get(i), s) -
                    get_element(a_dot_x, s) * spin_a[i] / square(rboyer)) *
                       dr[j] +
                   spin_a[i] * spin_a[j] / rboyer);
        if (i == j) {
          get_element(deriv_l->get(i + 1, j + 1), s) += den * rboyer;
        } else {  //  add den*epsilon^ijk a_k
          size_t k = (j + 1) % 3;
          if (k == i) {  // j+1 = i (cyclic), so choose minus sign
            ++k;
            k %= 3;  // and set k to be neither i nor j
            get_element(deriv_l->get(i + 1, j + 1), s) -= den * spin_a[k];
          } else {  // i+1 = j (cyclic), so choose plus sign
            get_element(deriv_l->get(i + 1, j + 1), s) += den * spin_a[k];
          }
        }
      }
    }

    for (size_t j = 0; j < 3; ++j) {
      for (size_t i = 0; i < 3; ++i) {
        temp_deriv_l[i, j] = get_element(deriv_l->get(i + 1, j + 1), s);
      }
    }

    for (size_t j = 0; j < 3; ++j) {
      for (size_t i = 0; i < 3; ++i) {
        get_element(deriv_l->get(i + 1, j + 1), s) = 0.;
        for (size_t k = 0; k < 3; ++k) {
          for (size_t m = 0; m < 3; ++m) {
            get_element(deriv_l->get(i + 1, j + 1), s) +=
                get_element(jacobian.get(k, i), s) *
                get_element(jacobian.get(m, j), s) *
                get_element(temp_deriv_l[k], m);
          }
          get_element(deriv_l->get(i + 1, j + 1), s) +=
              get_element(kerr_schild_l.get(k), s) *
              get_element(deriv_jacobian.get(k, i, j), s);
        }
      }
    }
  }

  std::cout << "this is deriv_l: " << *deriv_l << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse_squared,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::lapse_squared<DataType> /*meta*/) const noexcept {
  const auto& H = get(cache->get_var(internal_tags::H<DataType>{}));
  const auto& sph_kerr_schild_l_upper =
      cache->get_var(internal_tags::sph_kerr_schild_l_upper<DataType, Frame>{});
  get(*lapse_squared) =
      1.0 / (1.0 + 2.0 * square(sph_kerr_schild_l_upper.get(0)) * H);

  std::cout << "this is lapse_squared: " << *lapse_squared << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::Lapse<DataType> /*meta*/) const noexcept {
  const auto& lapse_squared =
      get(cache->get_var(internal_tags::lapse_squared<DataType>{}));
  get(*lapse) = sqrt(lapse_squared);

  std::cout << "this is lapse: " << *lapse << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> deriv_lapse_multiplier,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_lapse_multiplier<DataType> /*meta*/) const noexcept {
  const auto& lapse = get(cache->get_var(gr::Tags::Lapse<DataType>{}));
  const auto& lapse_squared =
      get(cache->get_var(internal_tags::lapse_squared<DataType>{}));
  get(*deriv_lapse_multiplier) =
      -square(null_vector_0_) * lapse * lapse_squared;

  std::cout << "this is the deriv_lapse_multiplier: " << *deriv_lapse_multiplier
            << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> shift_multiplier,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::shift_multiplier<DataType> /*meta*/) const noexcept {
  const auto& H = get(cache->get_var(internal_tags::H<DataType>{}));
  const auto& lapse_squared =
      get(cache->get_var(internal_tags::lapse_squared<DataType>{}));

  get(*shift_multiplier) = -2.0 * null_vector_0_ * H * lapse_squared;

  std::cout << "this is the shift_multiplier: " << *shift_multiplier << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> shift,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::Shift<3, Frame, DataType> /*meta*/) const noexcept {
  const auto& sph_kerr_schild_l_upper =
      cache->get_var(internal_tags::sph_kerr_schild_l_upper<DataType, Frame>{});
  const auto& shift_multiplier =
      get(cache->get_var(internal_tags::shift_multiplier<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    shift->get(i) = shift_multiplier * sph_kerr_schild_l_upper.get(i + 1);
  }

  std::cout << "this is the shift: " << *shift << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::iJ<DataType, 3, Frame>*> deriv_shift,
    const gsl::not_null<CachedBuffer*> cache,
    DerivShift<DataType, Frame> /*meta*/) const noexcept {
  const auto& H = get(cache->get_var(internal_tags::H<DataType>{}));
  const auto& sph_kerr_schild_l_upper =
      cache->get_var(internal_tags::sph_kerr_schild_l_upper<DataType, Frame>{});
  const auto& lapse_squared =
      get(cache->get_var(internal_tags::lapse_squared<DataType>{}));
  const auto& deriv_H =
      cache->get_var(internal_tags::deriv_H<DataType, Frame>{});
  const auto& deriv_l =
      cache->get_var(internal_tags::deriv_l<DataType, Frame>{});

  for (size_t m = 0; m < 3; ++m) {
    for (size_t i = 0; i < 3; ++i) {
      deriv_shift->get(m, i) =
          4.0 * cube(null_vector_0_) * H * sph_kerr_schild_l_upper.get(i + 1) *
              square(lapse_squared) * deriv_H.get(m) -
          2.0 * null_vector_0_ * lapse_squared *
              (sph_kerr_schild_l_upper.get(i + 1) * deriv_H.get(m) +
               H * deriv_l.get(m + 1, i + 1));
    }
  }
  std::cout << "this is deriv_shift: " << *deriv_shift << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> spatial_metric,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::SpatialMetric<3, Frame, DataType> /*meta*/) const noexcept {
  const auto& H = get(cache->get_var(internal_tags::H<DataType>{}));
  const auto& sph_kerr_schild_l_upper =
      cache->get_var(internal_tags::sph_kerr_schild_l_upper<DataType, Frame>{});
  const auto& jacobian =
      cache->get_var(internal_tags::jacobian<DataType, Frame>{});

  std::fill(spatial_metric->begin(), spatial_metric->end(), 0.);
  for (size_t i = 0; i < 3; ++i) {
    spatial_metric->get(i, i) = 1.;
    for (size_t j = i; j < 3; ++j) {  // Symmetry
      spatial_metric->get(i, j) += 2.0 * H *
                                   sph_kerr_schild_l_upper.get(i + 1) *
                                   sph_kerr_schild_l_upper.get(j + 1);
    }
  }

  for (size_t k = 0; k < 3; ++k) {
    for (size_t m = k; m < 3; ++m) {
      for (size_t i = 0; i < 3; ++i) {
        spatial_metric->get(k, m) += jacobian.get(i, k) * jacobian.get(i, m);
      }
    }
  }
  std::cout << "this is the spatial metric: " << *spatial_metric << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, 3, Frame>*> deriv_spatial_metric,
    const gsl::not_null<CachedBuffer*> cache,
    DerivSpatialMetric<DataType, Frame> /*meta*/) const noexcept {
  const auto& sph_kerr_schild_l_upper =
      cache->get_var(internal_tags::sph_kerr_schild_l_upper<DataType, Frame>{});
  const auto& deriv_H =
      cache->get_var(internal_tags::deriv_H<DataType, Frame>{});
  const auto& H = get(cache->get_var(internal_tags::H<DataType>{}));
  const auto& deriv_l =
      cache->get_var(internal_tags::deriv_l<DataType, Frame>{});

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {  // Symmetry
      for (size_t m = 0; m < 3; ++m) {
        deriv_spatial_metric->get(m, i, j) =
            2.0 * sph_kerr_schild_l_upper.get(i + 1) *
                sph_kerr_schild_l_upper.get(j + 1) * deriv_H.get(m) +
            2.0 * H *
                (sph_kerr_schild_l_upper.get(i + 1) *
                     deriv_l.get(m + 1, j + 1) +
                 sph_kerr_schild_l_upper.get(j + 1) *
                     deriv_l.get(m + 1, i + 1));
      }
    }
  }
  std::cout << "this is deriv_spatial_metric: " << *deriv_spatial_metric
            << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> dt_spatial_metric,
    const gsl::not_null<CachedBuffer*> /*cache*/,
    ::Tags::dt<gr::Tags::SpatialMetric<3, Frame, DataType>> /*meta*/)
    const noexcept {
  std::fill(dt_spatial_metric->begin(), dt_spatial_metric->end(), 0.);

  std::cout << "this is dt_spatial_metric: " << *dt_spatial_metric << "\n";
}

template <typename DataType, typename Frame>
SphKerrSchild::IntermediateVars<DataType, Frame>::IntermediateVars(
    const SphKerrSchild& solution,
    const tnsr::I<DataType, 3, Frame>& x) noexcept
    : CachedBuffer(get_size(::get<0>(x)), IntermediateComputer<DataType, Frame>(
                                              solution, x, null_vector_0_)) {}

template <typename DataType, typename Frame>
tnsr::i<DataType, 3, Frame>
SphKerrSchild::IntermediateVars<DataType, Frame>::get_var(
    DerivLapse<DataType, Frame> /*meta*/) noexcept {
  tnsr::i<DataType, 3, Frame> result{};
  const auto& deriv_H = get_var(internal_tags::deriv_H<DataType, Frame>{});
  const auto& deriv_lapse_multiplier =
      get(get_var(internal_tags::deriv_lapse_multiplier<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    result.get(i) = deriv_lapse_multiplier * deriv_H.get(i);
  }
  return result;
}

template <typename DataType, typename Frame>
Scalar<DataType> SphKerrSchild::IntermediateVars<DataType, Frame>::get_var(
    ::Tags::dt<gr::Tags::Lapse<DataType>> /*meta*/) noexcept {
  const auto& H = get(get_var(internal_tags::H<DataType>{}));
  return make_with_value<Scalar<DataType>>(H, 0.);
}

template <typename DataType, typename Frame>
tnsr::I<DataType, 3, Frame>
SphKerrSchild::IntermediateVars<DataType, Frame>::get_var(
    ::Tags::dt<gr::Tags::Shift<3, Frame, DataType>> /*meta*/) noexcept {
  const auto& H = get(get_var(internal_tags::H<DataType>()));
  return make_with_value<tnsr::I<DataType, 3, Frame>>(H, 0.);
}

template <typename DataType, typename Frame>
Scalar<DataType> SphKerrSchild::IntermediateVars<DataType, Frame>::get_var(
    gr::Tags::SqrtDetSpatialMetric<DataType> /*meta*/) noexcept {
  return Scalar<DataType>(1.0 / get(get_var(gr::Tags::Lapse<DataType>{})));
}

template <typename DataType, typename Frame>
tnsr::II<DataType, 3, Frame>
SphKerrSchild::IntermediateVars<DataType, Frame>::get_var(
    gr::Tags::InverseSpatialMetric<3, Frame, DataType> /*meta*/) noexcept {
  const auto& H = get(get_var(internal_tags::H<DataType>{}));
  const auto& lapse_squared =
      get(get_var(internal_tags::lapse_squared<DataType>{}));
  const auto& sph_kerr_schild_l_upper =
      get_var(internal_tags::sph_kerr_schild_l_upper<DataType, Frame>{});

  auto result = make_with_value<tnsr::II<DataType, 3, Frame>>(H, 0.);
  for (size_t i = 0; i < 3; ++i) {
    result.get(i, i) = 1.;
    for (size_t j = i; j < 3; ++j) {  // Symmetry
      result.get(i, j) -= 2.0 * H * lapse_squared *
                          sph_kerr_schild_l_upper.get(i + 1) *
                          sph_kerr_schild_l_upper.get(j + 1);
    }
  }

  return result;
}

template <typename DataType, typename Frame>
tnsr::ii<DataType, 3, Frame>
SphKerrSchild::IntermediateVars<DataType, Frame>::get_var(
    gr::Tags::ExtrinsicCurvature<3, Frame, DataType> /*meta*/) noexcept {
  return gr::extrinsic_curvature(
      get_var(gr::Tags::Lapse<DataType>{}),
      get_var(gr::Tags::Shift<3, Frame, DataType>{}),
      get_var(DerivShift<DataType, Frame>{}),
      get_var(gr::Tags::SpatialMetric<3, Frame, DataType>{}),
      get_var(::Tags::dt<gr::Tags::SpatialMetric<3, Frame, DataType>>{}),
      get_var(DerivSpatialMetric<DataType, Frame>{}));
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                \
  template class SphKerrSchild::IntermediateVars<DTYPE(data), FRAME(data)>; \
  template class SphKerrSchild::IntermediateComputer<DTYPE(data), FRAME(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector, double),
                        (::Frame::Inertial, ::Frame::Grid))
#undef INSTANTIATE
#undef DTYPE
#undef FRAME
}  // namespace gr::Solutions
