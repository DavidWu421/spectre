// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Ccz4/TagsDeclarations.hpp"
#include "Evolution/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"

namespace Ccz4 {
namespace Tags {
/*!
 * \brief The conformal factor that rescales the spatial metric
 *
 * \details If \f$\gamma_{ij}\f$ is the spatial metric, then we define
 * \f$\phi = (det(\gamma_{ij}))^{-1/6}\f$.
 */
template <typename DataType>
struct ConformalFactor : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief The conformally scaled spatial metric
 *
 * \details If \f$\phi\f$ is the conformal factor and \f$\gamma_{ij}\f$ is the
 * spatial metric, then we define
 * \f$\bar{\gamma}_{ij} = \phi^2 \gamma_{ij}\f$.
 */
template <size_t Dim, typename Frame, typename DataType>
using ConformalMetric =
    gr::Tags::Conformal<gr::Tags::SpatialMetric<Dim, Frame, DataType>, 2>;

/*!
 * \brief The conformally scaled inverse spatial metric
 *
 * \details If \f$\phi\f$ is the conformal factor and \f$\gamma^{ij}\f$ is the
 * inverse spatial metric, then we define
 * \f$\bar{\gamma}^{ij} = \phi^{-2} \gamma^{ij}\f$.
 */
template <size_t Dim, typename Frame, typename DataType>
using InverseConformalMetric =
    gr::Tags::Conformal<gr::Tags::InverseSpatialMetric<Dim, Frame, DataType>,
                        -2>;

/*!
 * \brief The natural log of the lapse
 */
template <typename DataType>
struct LogLapse : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief Auxiliary variable which is analytically the spatial derivative of the
 * natural log of the lapse
 *
 * \details If \f$ \alpha \f$ is the lapse, then we define
 * \f$A_i = \partial_i ln(\alpha) = \frac{\partial_i \alpha}{\alpha}\f$.
 */
template <size_t Dim, typename Frame, typename DataType>
struct FieldA : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

/*!
 * \brief Auxiliary variable which is analytically the spatial derivative of the
 * shift
 */
template <size_t Dim, typename Frame, typename DataType>
struct FieldB : db::SimpleTag {
  using type = tnsr::iJ<DataType, Dim, Frame>;
};

/*!
 * \brief Auxiliary variable which is analytically half the spatial derivative
 * of the conformal spatial metric
 *
 * \details If \f$\bar{\gamma}_{ij}\f$ is the conformal spatial metric, then we
 * define
 * \f$D_{kij} = \frac{1}{2} \partial_k \bar{\gamma}_{ij}\f$.
 */
template <size_t Dim, typename Frame, typename DataType>
struct FieldD : db::SimpleTag {
  using type = tnsr::ijj<DataType, Dim, Frame>;
};

/*!
 * \brief The natural log of the conformal factor
 */
template <typename DataType>
struct LogConformalFactor : db::SimpleTag {
  using type = Scalar<DataType>;
};

/*!
 * \brief Auxiliary variable which is analytically the spatial derivative of the
 * natural log of the conformal factor
 *
 * \details If \f$\phi\f$ is the conformal factor, then we define
 * \f$P_i = \partial_i ln(\phi) = \frac{\partial_i \phi}{\phi}\f$.
 */
template <size_t Dim, typename Frame, typename DataType>
struct FieldP : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
};

/*!
 * \brief The conformal spatial christoffel symbols of the second kind
 *
 * \details We define:
 * \f{align}
 *     \tilde{\Gamma}^k_{ij} &=
 *         \tilde{\gamma}^{kl} (D_{ijl} + D_{jil} - D_{lij})
 * \f}
 * where \f$\tilde{\gamma}^{ij}\f$ and \f$D_{ijk}\f$ are the inverse conformal
 * spatial metric and the CCZ4 auxiliary variable defined by
 * `Ccz4::Tags::InverseConformalMetric` and `Ccz4::Tags::FieldD`, respectively.
 */
template <size_t Dim, typename Frame, typename DataType>
struct ConformalChristoffelSecondKind : db::SimpleTag {
  using type = tnsr::Ijj<DataType, Dim, Frame>;
};

/*!
 * \brief The spatial christoffel symbols of the second kind
 *
 * \details We define:
 * \details Computes the christoffel symbols as:
 * \f{align}
 *     \Gamma^k_{ij} &= \tilde{\Gamma}^k_{ij} -
 *         \tilde{\gamma}^{kl} (\tilde{\gamma}_{jl} P_i +
 *                              \tilde{\gamma}_{il} P_j -
 *                              \tilde{\gamma}_{ij} P_l)
 * \f}
 * where \f$\tilde{\gamma}^{ij}\f$, \f$\tilde{\gamma}_{ij}\f$,
 * \f$\tilde{\Gamma}^k_{ij}\f$, and \f$P_i\f$ are the conformal spatial metric,
 * the inverse conformal spatial metric, the conformal spatial christoffel
 * symbols of the second kind, and the CCZ4 auxiliary variable defined by
 * `Ccz4::Tags::ConformalMetric`, `Ccz4::Tags::InverseConformalMetric`,
 * `Ccz4::Tags::ConformalChristoffelSecondKind`, and `Ccz4::Tags::FieldP`,
 * respectively.
 */
template <size_t Dim, typename Frame, typename DataType>
struct ChristoffelSecondKind : db::SimpleTag {
  using type = tnsr::Ijj<DataType, Dim, Frame>;
};
}  // namespace Tags

namespace OptionTags {
/*!
 * \ingroup OptionGroupsGroup
 * Groups option tags related to the Ccz4 evolution system.
 */
struct Group {
  static std::string name() noexcept { return "Ccz4"; }
  static constexpr Options::String help{
      "Options for the CCZ4 evolution system"};
  using group = evolution::OptionTags::SystemGroup;
};
}  // namespace OptionTags
}  // namespace Ccz4