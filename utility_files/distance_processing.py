# distance_processing.py

import numpy as np
from evcouplings.compare import DistanceMap
# import cupy as cp
def adjust_number(index):
    if index < 30:
        return index + 7
    elif 30 <= index <= 1180:
        return index + 6
    else:
        return index

def inverse_adjust_number(index):
    if index < 30:
        return index - 7
    elif 30 <= index <= 1180:
        return index - 6
    else:
        return index
def process_distance_matrix(dist_map, original_indices, initial_coef):
    """
    Process the distance matrix and filter out invalid positions based on a 5 Å threshold.

    Parameters:
        dist_map (DistanceMap): DistanceMap object containing the distance matrix.
        original_indices (np.ndarray): Indices of original features (mutations).
        initial_coef (np.ndarray): Initial coefficient values.

    Returns:
        thresholded_distances (np.ndarray): Thresholded distance matrix.
        valid_original_indices (np.ndarray): Indices of valid original features.
        subset_coef (np.ndarray): Subset of coefficients for valid features.
        valid_positions (np.ndarray): Valid positions in the distance matrix.
    """
    ### Step 1: Adjust Mutant Positions to Distance Map PDB Positions ###
    # Adjust positions to PDB distance map indices
    adjusted_positions = np.array([adjust_number(index) for index in original_indices])

    # Ensure positions are valid (within the distance matrix bounds)
    valid_positions = adjusted_positions[adjusted_positions < dist_map.dist_matrix.shape[0]]

    # Identify invalid positions
    invalid_positions = np.setdiff1d(adjusted_positions, valid_positions)

    ### Step 2: Subset the Distance Matrix for Valid Positions ###
    # Subset the distance matrix for valid positions
    subset_distance_matrix = dist_map.dist_matrix[np.ix_(valid_positions, valid_positions)]

    ### Step 3: Apply the 5 Å Threshold ###
    # Initialize thresholded distance matrix
    thresholded_distances = np.full(subset_distance_matrix.shape, np.inf)

    # Extract pairwise contacts from the distance map (threshold: 5 Å)
    contacts_df = dist_map.contacts(max_dist=5.0)

    # Extract pairwise indices and distances
    indices_i = contacts_df['i'].astype(int).values
    indices_j = contacts_df['j'].astype(int).values
    distances = contacts_df['dist'].values

    # Map valid indices to subset positions
    position_mapping = {pos: i for i, pos in enumerate(valid_positions)}
    mapped_indices_i = [position_mapping.get(idx, None) for idx in indices_i]
    mapped_indices_j = [position_mapping.get(idx, None) for idx in indices_j]

    # Remove invalid mappings
    mapped_indices_i = np.array([i for i in mapped_indices_i if i is not None])
    mapped_indices_j = np.array([j for j in mapped_indices_j if j is not None])

    # Apply valid distances to the thresholded distance matrix
    for i, j, dist in zip(mapped_indices_i, mapped_indices_j, distances):
        if dist <= 5.0:
            thresholded_distances[i, j] = dist
            thresholded_distances[j, i] = dist  # Ensure symmetry

    ### Step 4: Subset Coefficients ###
    # Map invalid positions back to original indices
    mapped_invalid_positions = np.array([inverse_adjust_number(pos) for pos in invalid_positions])

    # Drop invalid positions from original indices
    valid_original_indices = original_indices[~np.isin(original_indices, mapped_invalid_positions)]

    # Subset coefficients based on valid original indices
    subset_coef = initial_coef[np.isin(original_indices, valid_original_indices)]

    ### Return All Outputs ###
    return thresholded_distances, valid_original_indices, subset_coef, valid_positions



# def process_distance_matrix(dist_map, original_indices, initial_coef):
#     ### Step 1: Adjust Mutant Positions to Distance Map PDB Positions ###
#     adjusted_positions = np.array([adjust_number(index) for index in original_indices])

#     # Ensure adjusted_positions are within the bounds of the dist_info matrix
#     valid_positions = adjusted_positions[adjusted_positions < dist_map.dist_matrix.shape[0]]

#     # Find positions not valid
#     difference_positions = np.setdiff1d(adjusted_positions, valid_positions)

#     ### Step 2: Subset the Distance Matrix Using Valid Positions ###
#     subset_distance_matrix = dist_map.dist_matrix[np.ix_(valid_positions, valid_positions)]


#     ### Step 3: Subset Coefficients ###
#     mapped_invalid_positions = np.array([inverse_adjust_number(pos) for pos in difference_positions])

#     # Drop invalid positions from original_indices
#     valid_original_indices = original_indices[~np.isin(original_indices, mapped_invalid_positions)]

#     # Subset coefficients
#     subset_coef = initial_coef[np.isin(original_indices, valid_original_indices)]

#     return subset_distance_matrix, valid_original_indices, subset_coef, valid_positions

def subset_data(X, original_indices, valid_original_indices):
    # Subset X based on valid_original_indices
    X_subset = X[:, np.isin(original_indices, valid_original_indices)]
    return X_subset

def compute_scale_param(dist_info):
    return max(np.std(dist_info), 1)



