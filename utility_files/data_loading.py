# data_loading.py
# This script contains functions for loading phenotype, variant, and alignment data
# as well as precomputed feature matrices and labels for antibiotic resistance prediction.

import pandas as pd
import os
from evcouplings.align import Alignment
import numpy as np

def load_phenotype_data(paths, drug):
    """
    Load and concatenate phenotype CSV files, filtering for records with non-null phenotype labels
    for the specified antibiotic drug.
    """
    dfs = []
    for path in paths:
        df = pd.read_csv(path, low_memory=False)
        if drug in df.columns:
            df = df.dropna(subset=[drug])
        else:
            print(f"Warning: {drug} column not found in {path}")
        dfs.append(df)
    phenotype_data = pd.concat(dfs, ignore_index=True)
    print(f"Number of records in phenotype data: {len(phenotype_data)}")
    return phenotype_data

def load_who_catalog(who_catalog_path):
    """
    Load the WHO mutation catalog and extract relevant fields for mutation interpretation.
    """
    who_catalog = pd.read_excel(who_catalog_path, sheet_name='Catalogue_master_file', header=2)
    frequency_df = who_catalog[['drug', 'gene', 'mutation', 'variant', 'effect',
                                'Present_R', 'Present_S', 'Present_SOLO_R', 'FINAL CONFIDENCE GRADING']]
    return frequency_df

def load_variants(csv_file_path):
    """
    Load a CSV file containing mutation variants and return it as a DataFrame.
    """
    variants_df = pd.read_csv(csv_file_path)
    return variants_df

def load_alignment(file_path, alphabet='-actg'):
    """
    Load sequence alignment from file using evcouplings Alignment class.
    """
    return Alignment.from_file(open(file_path), alphabet=alphabet)

def load_feature_matrix_and_labels(gene_name):
    """
    Load precomputed feature matrix and labels from disk for a given gene.
    These files are expected to be saved in the 'feature_matrix_labels' directory.
    """
    file_dir = "/work/pi_annagreen_umass_edu/mahbuba/resistance-prediction/data/feature_matrix_labels"
    feature_matrix_file = f'{file_dir}/{gene_name}_feature_matrix.npy'
    labels_file = f'{file_dir}/{gene_name}_labels.npy'

    if os.path.exists(feature_matrix_file) and os.path.exists(labels_file):
        print(f"Loading feature matrix and labels for {gene_name} from disk.")
        feature_matrix = np.load(feature_matrix_file)
        labels = np.load(labels_file)
    else:
        print(f"Need to create feature matrix and labels for {gene_name}.")
        feature_matrix, labels = None, None  # Placeholder: to be filled if dynamic creation is needed.

    return feature_matrix, labels
