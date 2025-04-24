# data_preprocessing.py
# Utility functions for sequence preprocessing, feature encoding, and alignment slicing
# for resistance prediction models based on M. tuberculosis protein sequences.

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import random

def drop_identical_columns(X):
    """
    Remove duplicate columns from the input matrix X.
    Returns the matrix with unique columns and their original indices.
    """
    X_T = X.T
    _, unique_indices = np.unique(X_T, axis=0, return_index=True)
    unique_indices_sorted = np.sort(unique_indices)
    X_unique = X_T[unique_indices_sorted].T
    return X_unique, unique_indices_sorted

def drop_identical_sequences(X):
    """
    Remove duplicate sequences (rows) from the input matrix X.
    Returns the unique sequences and their indices.
    """
    _, unique_indices = np.unique(X, axis=0, return_index=True)
    X_unique = X[unique_indices]
    return X_unique, unique_indices

def encode_labels(labels):
    """
    Encode phenotype labels into binary values: 1 for "R" (resistant), 0 otherwise.
    """
    return [1 if label == "R" else 0 for label in labels]

def filter_nan_labels(labels, *arrays):
    """
    Remove entries with missing ('nan') or intermediate ('I') labels.
    Returns filtered labels and corresponding arrays (e.g., sequences).
    """
    valid_indices = [i for i, label in enumerate(labels) if label != 'nan' and label != 'I']
    filtered_labels = [labels[i] for i in valid_indices]
    filtered_arrays = [[array[i] for i in valid_indices] for array in arrays]
    return filtered_labels, filtered_arrays

def scale_features(X):
    """
    Apply standard scaling (zero mean, unit variance) to the feature matrix X.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def select_subset_alignment(alignment, start, end, reference_numbers):
    """
    Select columns from the alignment matrix between start and end reference positions.
    Uses provided reference_numbers for mapping.
    """
    try:
        start_index = np.where(reference_numbers == start)[0][0] if start in reference_numbers else 0
        end_index = np.where(reference_numbers == end)[0][0] if end in reference_numbers else alignment.matrix.shape[1] - 1
        start_index = max(0, min(start_index, alignment.matrix.shape[1] - 1))
        end_index = max(0, min(end_index, alignment.matrix.shape[1]))

        if start_index >= end_index:
            print(f"Error: Invalid range (start_index={start_index}, end_index={end_index}). Returning None.")
            return None

        selection = np.arange(start_index, end_index)
        return alignment.select(columns=selection)
    except IndexError as e:
        print(f"Error in select_subset_alignment: {e}")
        return None

def find_closest_index(reference_numbers, target_index):
    """
    Find the closest match to a target index within a window range.
    """
    if target_index in reference_numbers:
        return target_index
    for i in range(1, 11):
        if target_index - i in reference_numbers:
            return target_index - i
        elif target_index + i in reference_numbers:
            return target_index + i
    return None

def find_column_for_gene(reference_numbers, gene_start):
    """
    Identify the column index in the reference matrix for a gene start position.
    """
    result = np.where(reference_numbers == gene_start)
    return result[1][0] if result[0].size > 0 else None

def sort_gene_indices(reference_numbers, start_index, end_index, alignment):
    """
    Select subset alignment for a gene based on start and end coordinates.
    Adjusts using closest valid indices.
    """
    start_index = find_closest_index(reference_numbers, start_index)
    end_index = find_closest_index(reference_numbers, end_index)
    column_index = find_column_for_gene(reference_numbers, start_index)

    if column_index is not None:
        subset_alignment = select_subset_alignment(alignment, start_index, end_index, reference_numbers[:, column_index])
        print(f"Processed subset alignment for gene start {start_index} and end {end_index} in column {column_index}")
    else:
        print(f"Gene start index {start_index} not found in h37rv_numbers.")
        subset_alignment = None

    return subset_alignment, column_index, start_index, end_index

def isolate_sequences_with_phenotype(alignments, phenotype_data, filenames):
    """
    Filter alignment to retain only sequences present in phenotype dataset.
    """
    isolates_included = set(phenotype_data.Isolate_mapped)
    seqs_to_select = np.array([y for y, x in enumerate(filenames) if x in isolates_included])
    filtered_alignment = alignments.select(sequences=seqs_to_select)
    return filtered_alignment

def convert_to_onehot_with_reference(aa_seq, ref_aa):
    """
    Generate a one-hot difference encoding: 0 if amino acid matches reference, 1 otherwise.
    """
    return np.array([0 if aa == ref else 1 for aa, ref in zip(aa_seq, ref_aa)])

def encode_sequence(sequence, reference_length, h37rv_aa_str):
    """
    Apply ref-based one-hot encoding to an input amino acid sequence.
    """
    return convert_to_onehot_with_reference(str(sequence), str(h37rv_aa_str))

def get_aa_positions_by_gene(df, gene_name):
    """
    Extract sorted amino acid positions for a given gene from WHO mutation catalog dataframe.
    """
    gene_df = df[df['gene'] == gene_name]
    aa_positions = sorted(gene_df['aa_pos'].unique())
    return aa_positions

def seed_everything(seed=42):
    """
    Set random seed for reproducibility across numpy and random modules.
    """
    random.seed(seed)
    np.random.seed(seed)