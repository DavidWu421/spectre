// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Protocols/Metavariables.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/ObjectCenter.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain {
namespace {
template <size_t Dim>
void test_simple_tags() {
  TestHelpers::db::test_simple_tag<Tags::Domain<Dim>>("Domain");
  TestHelpers::db::test_simple_tag<Tags::InitialExtents<Dim>>("InitialExtents");
  TestHelpers::db::test_simple_tag<Tags::InitialRefinementLevels<Dim>>(
      "InitialRefinementLevels");
  TestHelpers::db::test_simple_tag<Tags::Element<Dim>>("Element");
  TestHelpers::db::test_simple_tag<Tags::Mesh<Dim>>("Mesh");
  TestHelpers::db::test_simple_tag<Tags::ElementMap<Dim>>(
      "ElementMap(Inertial)");
  TestHelpers::db::test_simple_tag<Tags::ElementMap<Dim, Frame::Grid>>(
      "ElementMap(Grid)");
  TestHelpers::db::test_simple_tag<Tags::Coordinates<Dim, Frame::Grid>>(
      "GridCoordinates");
  TestHelpers::db::test_simple_tag<
      Tags::Coordinates<Dim, Frame::ElementLogical>>(
      "ElementLogicalCoordinates");
  TestHelpers::db::test_simple_tag<Tags::Coordinates<Dim, Frame::Inertial>>(
      "InertialCoordinates");
  TestHelpers::db::test_simple_tag<
      Tags::InverseJacobian<Dim, Frame::ElementLogical, Frame::Inertial>>(
      "InverseJacobian(ElementLogical,Inertial)");
  TestHelpers::db::test_simple_tag<
      Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>>(
      "DetInvJacobian(ElementLogical,Inertial)");
  TestHelpers::db::test_simple_tag<Tags::InternalDirections<Dim>>(
      "InternalDirections");
  TestHelpers::db::test_simple_tag<Tags::BoundaryDirectionsInterior<Dim>>(
      "BoundaryDirectionsInterior");
  TestHelpers::db::test_simple_tag<Tags::BoundaryDirectionsExterior<Dim>>(
      "BoundaryDirectionsExterior");
  TestHelpers::db::test_simple_tag<Tags::Direction<Dim>>("Direction");
  TestHelpers::db::test_simple_tag<
      Tags::Jacobian<Dim, Frame::ElementLogical, Frame::Inertial>>(
      "Jacobian(ElementLogical,Inertial)");
}

void test_center_tags() {
  TestHelpers::db::test_base_tag<Tags::ObjectCenter<ObjectLabel::A>>(
      "ObjectCenter");
  TestHelpers::db::test_base_tag<Tags::ObjectCenter<ObjectLabel::B>>(
      "ObjectCenter");
  TestHelpers::db::test_simple_tag<Tags::ExcisionCenter<ObjectLabel::A>>(
      "CenterObjectA");
  TestHelpers::db::test_simple_tag<Tags::ExcisionCenter<ObjectLabel::B>>(
      "CenterObjectB");

  using Object = domain::creators::BinaryCompactObject::Object;

  const std::unique_ptr<DomainCreator<3>> domain_creator =
      std::make_unique<domain::creators::BinaryCompactObject>(
          Object{0.2, 5.0, 8.0, true, true}, Object{0.6, 4.0, -5.5, true, true},
          100.0, 500.0, 1_st, 5_st);

  const auto grid_center_A =
      Tags::ExcisionCenter<ObjectLabel::A>::create_from_options(domain_creator);
  const auto grid_center_B =
      Tags::ExcisionCenter<ObjectLabel::B>::create_from_options(domain_creator);

  CHECK(grid_center_A == tnsr::I<double, 3, Frame::Grid>{{8.0, 0.0, 0.0}});
  CHECK(grid_center_B == tnsr::I<double, 3, Frame::Grid>{{-5.5, 0.0, 0.0}});

  const std::unique_ptr<DomainCreator<3>> creator_no_excision =
      std::make_unique<domain::creators::Brick>(
          std::array{0.0, 0.0, 0.0}, std::array{1.0, 1.0, 1.0},
          std::array{0_st, 0_st, 0_st}, std::array{2_st, 2_st, 2_st},
          std::array{false, false, false});

  CHECK_THROWS_WITH(
      Tags::ExcisionCenter<ObjectLabel::B>::create_from_options(
          creator_no_excision),
      Catch::Contains(" is not in the domains excision spheres but is needed "
                      "to generate the ExcisionCenter"));
}

template <size_t Dim>
ElementMap<Dim, Frame::Grid> element_map();

template <>
ElementMap<1, Frame::Grid> element_map() {
  constexpr size_t dim = 1;
  const auto segment_ids = std::array<SegmentId, dim>({{SegmentId(2, 3)}});
  const CoordinateMaps::Affine first_map{-3.0, 8.7, 0.4, 5.5};
  const CoordinateMaps::Affine second_map{1.0, 8.0, -2.5, -1.0};
  const ElementId<dim> element_id(0, segment_ids);
  return ElementMap<dim, Frame::Grid>{
      element_id, make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
                      first_map, second_map)};
}

template <>
ElementMap<2, Frame::Grid> element_map() {
  constexpr size_t dim = 2;
  const auto segment_ids =
      std::array<SegmentId, dim>({{SegmentId(2, 3), SegmentId(1, 0)}});
  const CoordinateMaps::Rotation<dim> first_map(1.6);
  const CoordinateMaps::Rotation<dim> second_map(3.1);
  const ElementId<dim> element_id(0, segment_ids);
  return ElementMap<dim, Frame::Grid>{
      element_id, make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
                      first_map, second_map)};
}

template <>
ElementMap<3, Frame::Grid> element_map() {
  constexpr size_t dim = 3;
  const auto segment_ids = std::array<SegmentId, dim>(
      {{SegmentId(2, 3), SegmentId(1, 0), SegmentId(2, 1)}});
  const CoordinateMaps::Rotation<dim> first_map{M_PI_4, M_PI_4, M_PI_2};
  const CoordinateMaps::Rotation<dim> second_map{M_PI_4, M_PI_2, M_PI_2};
  const ElementId<dim> element_id(0, segment_ids);
  return ElementMap<dim, Frame::Grid>{
      element_id, make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
                      first_map, second_map)};
}

template <size_t Dim>
void test_compute_tags() {
  TestHelpers::db::test_compute_tag<Tags::InverseJacobianCompute<
      Tags::ElementMap<Dim>, Tags::Coordinates<Dim, Frame::ElementLogical>>>(
      "InverseJacobian(ElementLogical,Inertial)");
  TestHelpers::db::test_compute_tag<
      Tags::DetInvJacobianCompute<Dim, Frame::ElementLogical, Frame::Inertial>>(
      "DetInvJacobian(ElementLogical,Inertial)");
  TestHelpers::db::test_compute_tag<Tags::InternalDirectionsCompute<Dim>>(
      "InternalDirections");
  TestHelpers::db::test_compute_tag<
      Tags::BoundaryDirectionsInteriorCompute<Dim>>(
      "BoundaryDirectionsInterior");
  TestHelpers::db::test_compute_tag<
      Tags::BoundaryDirectionsExteriorCompute<Dim>>(
      "BoundaryDirectionsExterior");
  TestHelpers::db::test_compute_tag<Tags::MappedCoordinates<
      Tags::ElementMap<Dim>, Tags::Coordinates<Dim, Frame::ElementLogical>>>(
      "InertialCoordinates");
  TestHelpers::db::test_compute_tag<
      Tags::JacobianCompute<Dim, Frame::ElementLogical, Frame::Inertial>>(
      "Jacobian(ElementLogical,Inertial)");

  auto map = element_map<Dim>();
  const tnsr::I<DataVector, Dim, Frame::ElementLogical> logical_coords(
      make_array<Dim>(DataVector{-1.0, -0.5, 0.0, 0.5, 1.0}));
  const auto expected_inv_jacobian = map.inv_jacobian(logical_coords);
  const auto expected_jacobian = map.jacobian(logical_coords);

  const auto box = db::create<
      tmpl::list<Tags::ElementMap<Dim, Frame::Grid>,
                 Tags::Coordinates<Dim, Frame::ElementLogical>>,
      db::AddComputeTags<
          Tags::InverseJacobianCompute<
              Tags::ElementMap<Dim, Frame::Grid>,
              Tags::Coordinates<Dim, Frame::ElementLogical>>,
          Tags::DetInvJacobianCompute<Dim, Frame::ElementLogical, Frame::Grid>,
          Tags::JacobianCompute<Dim, Frame::ElementLogical, Frame::Grid>>>(
      std::move(map), logical_coords);
  CHECK_ITERABLE_APPROX(
      (db::get<Tags::InverseJacobian<Dim, Frame::ElementLogical, Frame::Grid>>(
          box)),
      expected_inv_jacobian);
  CHECK_ITERABLE_APPROX(
      (db::get<Tags::DetInvJacobian<Frame::ElementLogical, Frame::Grid>>(box)),
      determinant(expected_inv_jacobian));
  CHECK_ITERABLE_APPROX(
      (db::get<Tags::Jacobian<Dim, Frame::ElementLogical, Frame::Grid>>(box)),
      expected_jacobian);
}

SPECTRE_TEST_CASE("Unit.Domain.Tags", "[Unit][Domain]") {
  test_simple_tags<1>();
  test_simple_tags<2>();
  test_simple_tags<3>();

  test_center_tags();

  test_compute_tags<1>();
  test_compute_tags<2>();
  test_compute_tags<3>();
}
}  // namespace
}  // namespace domain
