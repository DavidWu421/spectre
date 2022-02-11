// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <pup.h>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Solutions.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Tags {
template <typename Tag>
struct dt;
}  // namespace Tags
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace gr {
namespace Solutions {

class SphKerrSchild : public AnalyticSolution<3_st>,
                      public MarkAsAnalyticSolution {
 public:
  struct Mass {
    using type = double;
    static constexpr Options::String help = {"Mass of the black hole"};
    static type lower_bound() { return 0.; }
  };
  struct Spin {
    using type = std::array<double, volume_dim>;
    static constexpr Options::String help = {
        "The [x,y,z] dimensionless spin of the black hole"};
  };
  struct Center {
    using type = std::array<double, volume_dim>;
    static constexpr Options::String help = {
        "The [x,y,z] center of the black hole"};
  };
  using options = tmpl::list<Mass, Spin, Center>;
  static constexpr Options::String help{
      "Black hole in SphKerr-Schild coordinates"};

  SphKerrSchild(double mass, Spin::type dimensionless_spin, Center::type center,
                const Options::Context& context = {});

  explicit SphKerrSchild(CkMigrateMessage* /*unused*/) {}

  SphKerrSchild() = default;
  SphKerrSchild(const SphKerrSchild& /*rhs*/) = default;
  SphKerrSchild& operator=(const SphKerrSchild& /*rhs*/) = default;
  SphKerrSchild(SphKerrSchild&& /*rhs*/) = default;
  SphKerrSchild& operator=(SphKerrSchild&& /*rhs*/) = default;
  ~SphKerrSchild() = default;

  template <typename DataType, typename Frame, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, volume_dim, Frame>& x, double /*t*/,
      tmpl::list<Tags...> /*meta*/) const {
    static_assert(
        tmpl2::flat_all_v<tmpl::list_contains_v<
            tmpl::push_back<
                tags<DataType, Frame>,
                gr::Tags::DerivDetSpatialMetric<3, Frame, DataType>>,
            Tags>...>,
        "At least one of the requested tags is not supported. The requested "
        "tags are listed as template parameters of the `variables` function.");
    IntermediateVars<DataType, Frame> cache(get_size(*x.begin()));
    IntermediateComputer<DataType, Frame> computer(*this, x);
    return {cache.get_var(computer, Tags{})...};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  SPECTRE_ALWAYS_INLINE double mass() const { return mass_; }
  SPECTRE_ALWAYS_INLINE const std::array<double, volume_dim>& center() const {
    return center_;
  }
  SPECTRE_ALWAYS_INLINE const std::array<double, volume_dim>&
  dimensionless_spin() const {
    return dimensionless_spin_;
  }

  struct internal_tags {
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using x_sph_minus_center = ::Tags::TempI<0, 3, Frame, DataType>;
    template <typename DataType>
    using r_squared = ::Tags::TempScalar<1, DataType>;
    template <typename DataType>
    using r = ::Tags::TempScalar<2, DataType>;
    template <typename DataType>
    using rho = ::Tags::TempScalar<3, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using matrix_F = ::Tags::TempIj<4, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using matrix_P = ::Tags::TempIj<5, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using jacobian = ::Tags::TempIj<6, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using matrix_D = ::Tags::TempIj<7, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using matrix_C = ::Tags::TempIj<8, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using deriv_jacobian = ::Tags::TempiJk<9, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using matrix_Q = ::Tags::TempIj<10, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using matrix_G1 = ::Tags::TempIj<11, 3, Frame, DataType>;
    template <typename DataType>
    using a_dot_x = ::Tags::TempScalar<12, DataType>;
    template <typename DataType>
    using s_number = ::Tags::TempScalar<13, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using matrix_G2 = ::Tags::TempIj<14, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using G1_dot_x = ::Tags::TempI<15, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using G2_dot_x = ::Tags::Tempi<16, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using inv_jacobian = ::Tags::TempIj<17, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using matrix_E1 = ::Tags::TempIj<18, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using matrix_E2 = ::Tags::TempIj<19, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using deriv_inv_jacobian = ::Tags::TempiJk<20, 3, Frame, DataType>;
    template <typename DataType>
    using H = ::Tags::TempScalar<22, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using x_kerr_schild = ::Tags::TempI<23, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using a_cross_x = ::Tags::TempI<24, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using kerr_schild_l = ::Tags::TempI<25, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using sph_kerr_schild_l_lower = ::Tags::Tempi<26, 4, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using sph_kerr_schild_l_upper = ::Tags::TempI<27, 4, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using deriv_H = ::Tags::TempI<28, 4, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using deriv_l = ::Tags::TempIj<28, 4, Frame, DataType>;
  };

  template <typename DataType, typename Frame = ::Frame::Inertial>
  using CachedBuffer = CachedTempBuffer<
      internal_tags::x_sph_minus_center<DataType, Frame>,
      internal_tags::r_squared<DataType>, internal_tags::r<DataType>,
      internal_tags::rho<DataType>, internal_tags::matrix_F<DataType, Frame>,
      internal_tags::matrix_P<DataType, Frame>,
      internal_tags::jacobian<DataType, Frame>,
      internal_tags::matrix_D<DataType, Frame>,
      internal_tags::matrix_C<DataType, Frame>,
      internal_tags::deriv_jacobian<DataType, Frame>,
      internal_tags::matrix_Q<DataType, Frame>,
      internal_tags::matrix_G1<DataType, Frame>,
      internal_tags::a_dot_x<DataType>, internal_tags::s_number<DataType>,
      internal_tags::matrix_G2<DataType, Frame>,
      internal_tags::G1_dot_x<DataType, Frame>,
      internal_tags::G2_dot_x<DataType, Frame>,
      internal_tags::inv_jacobian<DataType, Frame>,
      internal_tags::matrix_E1<DataType, Frame>,
      internal_tags::matrix_E2<DataType, Frame>,
      internal_tags::deriv_inv_jacobian<DataType, Frame>,
      internal_tags::H<DataType>, internal_tags::x_kerr_schild<DataType, Frame>,
      internal_tags::a_cross_x<DataType, Frame>,
      internal_tags::kerr_schild_l<DataType, Frame>,
      internal_tags::sph_kerr_schild_l_lower<DataType, Frame>,
      internal_tags::sph_kerr_schild_l_upper<DataType, Frame>,
      internal_tags::deriv_H<DataType, Frame>,
      internal_tags::deriv_l<DataType, Frame>>;

  template <typename DataType, typename Frame = ::Frame::Inertial>
  class IntermediateComputer {
   public:
    using CachedBuffer = SphKerrSchild::CachedBuffer<DataType, Frame>;

    IntermediateComputer(const SphKerrSchild& solution,
                         const tnsr::I<DataType, 3, Frame>& x);

    void operator()(
        gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_sph_minus_center,
        gsl::not_null<CachedBuffer*> /*cache*/,
        internal_tags::x_sph_minus_center<DataType, Frame> /*meta*/) const;

    void operator()(gsl::not_null<Scalar<DataType>*> r_squared,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::r_squared<DataType> /*meta*/) const;

    void operator()(gsl::not_null<Scalar<DataType>*> r,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::r<DataType> /*meta*/) const;

    void operator()(gsl::not_null<Scalar<DataType>*> rho,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::rho<DataType> /*meta*/) const;

    void operator()(gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_F,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::matrix_F<DataType, Frame> /*meta*/) const;

    void operator()(gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_P,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::matrix_P<DataType, Frame> /*meta*/) const;

    void operator()(gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> jacobian,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::jacobian<DataType, Frame> /*meta*/) const;

    void operator()(gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_D,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::matrix_D<DataType, Frame> /*meta*/) const;

    void operator()(gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_C,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::matrix_C<DataType, Frame> /*meta*/) const;

    void operator()(
        gsl::not_null<tnsr::iJk<DataType, 3, Frame>*> deriv_jacobian,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::deriv_jacobian<DataType, Frame> /*meta*/) const;

    void operator()(gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_Q,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::matrix_Q<DataType, Frame> /*meta*/) const;

    void operator()(gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_G1,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::matrix_G1<DataType, Frame> /*meta*/) const;

    void operator()(gsl::not_null<Scalar<DataType>*> a_dot_x,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::a_dot_x<DataType> /*meta*/) const;

    void operator()(gsl::not_null<Scalar<DataType>*> s_number,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::s_number<DataType> /*meta*/) const;

    void operator()(gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_G2,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::matrix_G2<DataType, Frame> /*meta*/) const;

    void operator()(gsl::not_null<tnsr::I<DataType, 3, Frame>*> G1_dot_x,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::G1_dot_x<DataType, Frame> /*meta*/) const;

    void operator()(gsl::not_null<tnsr::i<DataType, 3, Frame>*> G2_dot_x,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::G2_dot_x<DataType, Frame> /*meta*/) const;

    void operator()(
        gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> inv_jacobian,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::inv_jacobian<DataType, Frame> /*meta*/) const;

    void operator()(gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_E1,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::matrix_E1<DataType, Frame> /*meta*/) const;

    void operator()(gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_E2,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::matrix_E2<DataType, Frame> /*meta*/) const;

    void operator()(
        gsl::not_null<tnsr::iJk<DataType, 3, Frame>*> deriv_inv_jacobian,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::deriv_inv_jacobian<DataType, Frame> /*meta*/) const;

    void operator()(gsl::not_null<Scalar<DataType>*> H,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::H<DataType> /*meta*/) const;

    void operator()(
        gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_kerr_schild,
        gsl::not_null<CachedBuffer*> /*cache*/,
        internal_tags::x_kerr_schild<DataType, Frame> /*meta*/) const;

    void operator()(gsl::not_null<tnsr::I<DataType, 3, Frame>*> a_cross_x,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::a_cross_x<DataType, Frame> /*meta*/) const;

    void operator()(
        gsl::not_null<tnsr::I<DataType, 3, Frame>*> kerr_schild_l,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::kerr_schild_l<DataType, Frame> /*meta*/) const;

    void operator()(
        gsl::not_null<tnsr::i<DataType, 4, Frame>*> sph_kerr_schild_l_lower,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::sph_kerr_schild_l_lower<DataType, Frame> /*meta*/) const;

    void operator()(
        gsl::not_null<tnsr::I<DataType, 4, Frame>*> sph_kerr_schild_l_upper,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::sph_kerr_schild_l_upper<DataType, Frame> /*meta*/) const;

    void operator()(gsl::not_null<tnsr::I<DataType, 4, Frame>*> deriv_H,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::deriv_H<DataType, Frame> /*meta*/) const;

    void operator()(gsl::not_null<tnsr::Ij<DataType, 4, Frame>*> deriv_l,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::deriv_l<DataType, Frame> /*meta*/) const;

   private:
    const SphKerrSchild& solution_;
    const tnsr::I<DataType, 3, Frame>& x_;
    // Here null_vector_0 is simply -1, but if you have a boosted solution,
    // then null_vector_0 can be something different, so we leave it coded
    // in instead of eliminating it.
    static constexpr double null_vector_0_ = -1.0;
  };

  template <typename DataType, typename Frame = ::Frame::Inertial>
  class IntermediateVars : public CachedBuffer<DataType, Frame> {
   public:
    using CachedBuffer = SphKerrSchild::CachedBuffer<DataType, Frame>;
    using CachedBuffer::CachedBuffer;
    using CachedBuffer::get_var;

    tnsr::i<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        DerivLapse<DataType, Frame> /*meta*/);

    Scalar<DataType> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        ::Tags::dt<gr::Tags::Lapse<DataType>> /*meta*/);

    tnsr::I<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        ::Tags::dt<gr::Tags::Shift<3, Frame, DataType>> /*meta*/);

    Scalar<DataType> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::SqrtDetSpatialMetric<DataType> /*meta*/);

    tnsr::i<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::DerivDetSpatialMetric<3, Frame, DataType> /*meta*/);

    tnsr::II<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::InverseSpatialMetric<3, Frame, DataType> /*meta*/);

    tnsr::ii<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::ExtrinsicCurvature<3, Frame, DataType> /*meta*/);

   private:
    // Here null_vector_0 is simply -1, but if you have a boosted solution,
    // then null_vector_0 can be something different, so we leave it coded
    // in instead of eliminating it.
    static constexpr double null_vector_0_ = -1.0;
  };

 private:
  double mass_{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, volume_dim> dimensionless_spin_ =
      make_array<volume_dim>(std::numeric_limits<double>::signaling_NaN());
  std::array<double, volume_dim> center_ =
      make_array<volume_dim>(std::numeric_limits<double>::signaling_NaN());
};

SPECTRE_ALWAYS_INLINE bool operator==(const SphKerrSchild& lhs,
                                      const SphKerrSchild& rhs) {
  return lhs.mass() == rhs.mass() and
         lhs.dimensionless_spin() == rhs.dimensionless_spin() and
         lhs.center() == rhs.center();
}

SPECTRE_ALWAYS_INLINE bool operator!=(const SphKerrSchild& lhs,
                                      const SphKerrSchild& rhs) {
  return not(lhs == rhs);
}
}  // namespace Solutions
}  // namespace gr
