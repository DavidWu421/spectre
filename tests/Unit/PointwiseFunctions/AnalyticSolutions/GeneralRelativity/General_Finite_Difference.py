# Distributed under the MIT License.
# See LICENSE.txt for details.

#CHECKS THE DERIVATIVE OF FUNCTION F WITH FINITE DIFFERENCE OF SOME
# SMALL PERTURBATION OF F

import numpy as np

def check_finite_difference_rank1(input_vector, perturbed_input_vectors,
     perturbation):
    #parameters:
        #input_vector: np.array of size(# of param of f): The function of
            # interest evaluated at the coordinates of interest.
        #Can also be one of the vectors that make up a higher dimensional
            # tensor

        #perturbed_input_vectors: rank 2 Tensor expressed as a np.array of
            # dim(# of param of f)**2: The function of interest evaluated
            # at the coordinates of interest plus some small perturbation.
        # Need a vector for a perturbation in each direction, so it is
        # a rank two tensor. Can also be a tensor composed of vectors of
        # a small perturbation that make up a higher dimensional
        # perturbed tensor

        #perturbation: np.array of size(# of param of f): Size of the
            # perturbation for the parameters of the function whose
            # derivative is being taken (for f(x,y,z), dx=dy=dz).

    input_vector = input_vector.tolist()
    perturbed_input_vectors = perturbed_input_vectors.tolist()
    perturbation = perturbation.tolist()
    derivative_tensor = []
    for i in range(len(input_vector)):
        dimension_1 = []
        for j in range(len(input_vector)):
            derivative_tensor_indexed_value = (perturbed_input_vectors[i][j] -
                input_vector[i])/perturbation[i]
            dimension_1.append(derivative_tensor_indexed_value)
        derivative_tensor.append(dimension_1)
    derivative_tensor = np.array(derivative_tensor)
    return derivative_tensor



def check_finite_difference_rank3(input_matrix, perturbed_input_matrix,
     perturbation):
    '''
    #parameters:
        #input_matrix: np.array of size(# of param of f): The function of
            # interest evaluated at the coordinates of interest.
        #Can also be one of the vectors that make up a higher dimensional
            # tensor

        #perturbed_input_matrix: rank 3 Tensor expressed as a np.array of
            # dim(# of param of f)**2: The function of interest evaluated
            # at the coordinates of interest plus some small perturbation.
        # Need a vector for a perturbation in each direction, so it is
        # a rank two tensor. Can also be a tensor composed of vectors of
        # a small perturbation that make up a higher dimensional
        # perturbed tensor

        #perturbation: np.array of size(# of param of f): Size of the
            # perturbation for the parameters of the function whose
            # derivative is being taken (for f(x,y,z), dx=dy=dz).
    Returns
    -------
    Type:
        rank 3 tensor
    '''

    length = len(perturbed_input_matrix)
    derivative_tensor = []
    for i in range(length):
        # Each entry is the derivative of the function
        # in the ith direction
        dimension_2 = []
        for j in range(length):
            dimension_1 = []
            for k in range(length):
                derivative_tensor_indexed_value = (
                    perturbed_input_matrix[j][i][k] -
                    input_matrix[j][i]) / (perturbation[k])
                dimension_1.append(derivative_tensor_indexed_value)
            dimension_2.append(dimension_1)
        derivative_tensor.append(dimension_2)
    derivative_tensor = np.array(derivative_tensor)
    return derivative_tensor
