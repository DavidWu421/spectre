// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <boost/program_options.hpp>
#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Helpers.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Printf.hpp"

#include <iostream>
#include <map>

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

std::tuple<size_t, size_t, std::array<int, 3>>
compute_expected_connectivity_length(const h5::VolumeData& volume_file,
                                     const size_t& single_obs_id,
                                     const size_t total_number_of_elements) {
  size_t expected_connectivity_length = 0;
  size_t expected_number_of_grid_points = 0;
  for (size_t i = 0; i < total_number_of_elements; ++i) {
    auto extents = volume_file.get_extents(single_obs_id);
    expected_connectivity_length +=
        (extents[i][0] - 1) * (extents[i][1] - 1) * (extents[i][2] - 1) * 8;
    expected_number_of_grid_points +=
        extents[i][0] * extents[i][1] * extents[i][2];
  }

  std::string grid_name_string = volume_file.get_grid_names(single_obs_id)[0];
  std::array<int, 3> h_ref_array;
  size_t h_ref_previous_start_position = 0;
  for (size_t i = 0; i < 3; ++i) {
    size_t h_ref_start_position =
        grid_name_string.find("L", h_ref_previous_start_position + 1);
    size_t h_ref_end_position =
        grid_name_string.find("I", h_ref_start_position);
    int h_ref = std::stoi(
        grid_name_string.substr(h_ref_start_position + 1,
                                h_ref_end_position - h_ref_start_position - 1));
    h_ref_array[i] = h_ref;
    h_ref_previous_start_position = h_ref_start_position;
  }
  expected_connectivity_length +=
      ((pow(2, h_ref_array[0] + 1) - 1) * (pow(2, h_ref_array[1] + 1) - 1) *
           (pow(2, h_ref_array[2] + 1) - 1) -
       total_number_of_elements) *
      8;

  return std::tuple{expected_connectivity_length,
                    expected_number_of_grid_points, h_ref_array};
}

std::pair<Mesh<3>, std::string> generate_element_properties(
    const h5::VolumeData& volume_file, const size_t& single_obs_id,
    const size_t element_number) {
  // MAKE SURE TO MAKE THE DIM GENERAL
  auto bases = volume_file.get_bases(single_obs_id);
  std::array<Spectral::Basis, 3> basis_array = {
      Spectral::to_basis(bases[element_number][0]),
      Spectral::to_basis(bases[0][1]),
      Spectral::to_basis(bases[element_number][2])};

  auto quadratures = volume_file.get_quadratures(single_obs_id);
  std::array<Spectral::Quadrature, 3> quadrature_array = {
      Spectral::to_quadrature(quadratures[element_number][0]),
      Spectral::to_quadrature(quadratures[element_number][1]),
      Spectral::to_quadrature(quadratures[element_number][2])};

  auto extents = volume_file.get_extents(single_obs_id);
  std::array<size_t, 3> extents_array = {extents[element_number][0],
                                         extents[element_number][1],
                                         extents[element_number][2]};

  std::pair<Mesh<3>, std::string> element_properties(
      Mesh<3>{extents_array, basis_array, quadrature_array},
      volume_file.get_grid_names(single_obs_id)[element_number]);

  return element_properties;
}

// Compute Block-logical coordinates
const tnsr::I<DataVector, 3, Frame::BlockLogical> generate_block_logical_coords(
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& new_coords,
    const std::string& element_id,
    const std::array<int, 3>& h_refinement_array) {
  size_t grid_points_x_start_position = 0;
  tnsr::I<DataVector, 3, Frame::BlockLogical> block_logical_coords{
      new_coords.get(0).size(), 0.};
  for (size_t i = 0; i < 3; ++i) {
    double number_of_elements = 0;
    number_of_elements = pow(2, h_refinement_array[i]);
    size_t grid_points_start_position =
        element_id.find("I", grid_points_x_start_position + 1);
    size_t grid_points_end_position =
        element_id.find(",", grid_points_start_position);
    if (i == 2) {
      grid_points_end_position =
          element_id.find(")", grid_points_start_position);
    }
    int element_index = std::stoi(element_id.substr(
        grid_points_start_position + 1,
        grid_points_end_position - grid_points_start_position - 1));
    double shift = (-1 + (2 * element_index + 1) / number_of_elements);
    DataVector block_logical =
        1 / number_of_elements * new_coords.get(i) + shift;
    block_logical_coords.get(i) = block_logical;
    grid_points_x_start_position = grid_points_start_position;
  }

  return block_logical_coords;
}

// Functions that compute the connectivity within the block
std::map<std::array<double, 3>, size_t> generate_grid_point_map(
    const tnsr::I<DataVector, 3, Frame::BlockLogical>& coord_data) {
  std::map<std::array<double, 3>, size_t> grid_point_map;
  DataVector coord_data_x = coord_data.get(0);
  DataVector coord_data_y = coord_data.get(1);
  DataVector coord_data_z = coord_data.get(2);
  for (size_t i = 0; i < coord_data_x.size(); ++i) {
    std::array<double, 3> coord_data_point{coord_data_x[i], coord_data_y[i],
                                           coord_data_z[i]};
    grid_point_map.insert(
        std::pair<std::array<double, 3>, size_t>(coord_data_point, i));
  }
  return grid_point_map;
}

std::vector<double> sort_and_order(DataVector& data_vector) {
  std::vector<double> ordered_coords;
  std::vector<double> data_list;
  for (size_t i = 0; i < data_vector.size(); ++i) {
    data_list.emplace_back(data_vector[i]);
  }
  sort(data_list.begin(), data_list.end());
  ordered_coords.push_back(data_list[0]);
  for (size_t i = 1; i < data_list.size(); ++i) {
    if (data_list[i] == ordered_coords.end()[-1]) {
      continue;
    } else {
      ordered_coords.push_back(data_list[i]);
    }
  }
  return ordered_coords;
}

std::vector<std::array<double, 3>> build_connectivity_by_element(
    std::vector<double>& sorted_x, std::vector<double>& sorted_y,
    std::vector<double>& sorted_z) {
  std::vector<std::array<double, 3>> connectivity_as_arrays;
  for (size_t k = 0; k < sorted_z.size() - 1; ++k) {
    for (size_t j = 0; j < sorted_y.size() - 1; ++j) {
      for (size_t i = 0; i < sorted_x.size() - 1; ++i) {
        connectivity_as_arrays.insert(
            connectivity_as_arrays.end(),
            {std::array<double, 3>{sorted_x[i], sorted_y[j], sorted_z[k]},
             std::array<double, 3>{sorted_x[i + 1], sorted_y[j], sorted_z[k]},
             std::array<double, 3>{sorted_x[i + 1], sorted_y[j + 1],
                                   sorted_z[k]},
             std::array<double, 3>{sorted_x[i], sorted_y[j + 1], sorted_z[k]},
             std::array<double, 3>{sorted_x[i], sorted_y[j], sorted_z[k + 1]},
             std::array<double, 3>{sorted_x[i + 1], sorted_y[j],
                                   sorted_z[k + 1]},
             std::array<double, 3>{sorted_x[i + 1], sorted_y[j + 1],
                                   sorted_z[k + 1]},
             std::array<double, 3>{sorted_x[i], sorted_y[j + 1],
                                   sorted_z[k + 1]}});
      }
    }
  }
  return connectivity_as_arrays;
}

void generate_new_connectivity(
    tnsr::I<DataVector, 3, Frame::BlockLogical>& block_logical_coords,
    std::vector<int>& new_connectivity) {
  std::map<std::array<double, 3>, size_t> grid_point_map =
      generate_grid_point_map(block_logical_coords);
  DataVector logical_coords_x = block_logical_coords.get(0);
  DataVector logical_coords_y = block_logical_coords.get(1);
  DataVector logical_coords_z = block_logical_coords.get(2);
  std::vector<double> ordered_x = sort_and_order(logical_coords_x);
  std::vector<double> ordered_y = sort_and_order(logical_coords_y);
  std::vector<double> ordered_z = sort_and_order(logical_coords_z);
  std::vector<std::array<double, 3>> connectivity_of_arrays =
      build_connectivity_by_element(ordered_x, ordered_y, ordered_z);

  for (const std::array<double, 3>& it : connectivity_of_arrays) {
    new_connectivity.push_back(grid_point_map[it]);
  }
}

void block_connectivity(const std::string& file_name,
                        const std::string& subfile_name) {
  h5::H5File<h5::AccessType::ReadWrite> data_file(file_name, true);
  const auto& volume_file = data_file.get<h5::VolumeData>("/" + subfile_name);
  auto observation_ids = volume_file.list_observation_ids();
  const auto& single_obs_id =
      observation_ids[0];  // Hard coded for only 1 observation ID
  size_t total_number_of_elements =
      volume_file.get_bases(single_obs_id)
          .size();  // Check this using grid_names for faster parsing?

  auto [expected_connectivity_length, expected_number_of_grid_points,
        h_refinement_array] =
      compute_expected_connectivity_length(volume_file, single_obs_id,
                                           total_number_of_elements);

  tnsr::I<DataVector, 3, Frame::BlockLogical> total_block_logical_coords{
      expected_number_of_grid_points, 0.};

  size_t offset = 0;
  for (size_t i = 0; i < volume_file.get_bases(single_obs_id).size(); ++i) {
    auto [element_mesh, element_id] =
        generate_element_properties(volume_file, single_obs_id, i);
    auto block_logical_coords = generate_block_logical_coords(
        logical_coordinates(element_mesh), element_id, h_refinement_array);
    for (size_t j = 0 + offset;
         j < (block_logical_coords.get(0).size() + offset); ++j) {
      total_block_logical_coords.get(0)[j] =
          block_logical_coords.get(0)[j - offset];
      total_block_logical_coords.get(1)[j] =
          block_logical_coords.get(1)[j - offset];
      total_block_logical_coords.get(2)[j] =
          block_logical_coords.get(2)[j - offset];
    }
    offset += block_logical_coords.get(0).size();
  }
  std::vector<int> new_connectivity;
  new_connectivity.reserve(expected_number_of_grid_points);
  generate_new_connectivity(total_block_logical_coords, new_connectivity);
  std::cout << new_connectivity << '\n';
}

// def extend_connectivity(input_file):

//     if sys.version_info < (3, 0):
//         logging.warning("You are attempting to run this script with "
//                         "python 2, which is deprecated. GenerateXdmf.py might
//                         " "hang or run very slowly using python 2. Please use
//                         " "python 3 instead.")

//     with h5py.File(input_file, 'r+') as dataset:
//         subfile_keys = list(dataset.keys())
//         subfile_name = subfile_keys[0]
//         data = dataset[subfile_name]
//         observation_keys = list(data.keys())
//         for i in range(len(observation_keys)):
//             print("Starting observation key number " + str(i))  ############
//             data_y = np.array(
//                 dataset[subfile_name + "/" + observation_keys[i] +
//                         "/InertialCoordinates_y"])
//             data_x = np.array(
//                 dataset[subfile_name + "/" + observation_keys[i] +
//                         "/InertialCoordinates_x"])
//             data_z = np.array(
//                 dataset[subfile_name + "/" + observation_keys[i] +
//                         "/InertialCoordinates_z"])
//             new_connect = generate_new_connectivity(data_x, data_y, data_z)
//             del dataset[subfile_name + "/" + observation_keys[i] +
//                         "/connectivity"]
//             dataset.create_dataset(subfile_name + "/" + observation_keys[i] +
//                                    "/connectivity",
//                                    data=new_connect)
//         dataset.close()

int main(int argc, char** argv) {
  boost::program_options::positional_options_description pos_desc;

  boost::program_options::options_description desc("Options");
  desc.add_options()("help,h,", "show this help message")(
      "file_name", boost::program_options::value<std::string>()->required(),
      "name of the file")(
      "subfile_name", boost::program_options::value<std::string>()->required(),
      "subfile name of the volume file in the H5 file (omit file "
      "extension)");

  boost::program_options::variables_map vars;

  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
          .positional(pos_desc)
          .options(desc)
          .run(),
      vars);

  if (vars.count("help") != 0u or vars.count("file_name") == 0u or
      vars.count("subfile_name") == 0u) {
    Parallel::printf("%s\n", desc);
    return 0;
  }

  block_connectivity(vars["file_name"].as<std::string>(),
                     vars["subfile_name"].as<std::string>());
}
