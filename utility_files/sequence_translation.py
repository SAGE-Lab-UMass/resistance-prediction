from Bio.Data import CodonTable
import re
import numpy as np

def translate_sequence_with_gaps(dna_seq, table="Standard", to_stop=False, handle_stops='remove', ref_protein_length=None):
    """
    Translates a DNA sequence into a protein sequence, handling gaps and ambiguous bases.

    Args:
        dna_seq (str): Input DNA sequence (may include '-' gaps).
        table (str): Codon table to use for translation.
        to_stop (bool): If True, stop translation at first stop codon.
        handle_stops (str): One of {'remove', 'replace', '*'} to handle stop codons.
        ref_protein_length (int, optional): Reference length to detect frameshifts.

    Returns:
        protein_str (str): Translated protein sequence with gaps handled.
        frameshift_mutations (bool): True if frameshift or early stop codon is detected.
    """
    codon_table = CodonTable.unambiguous_dna_by_name[table]
    standard_table = codon_table.forward_table
    stop_codons = codon_table.stop_codons

    dna_seq = dna_seq.upper()
    protein_seq = []
    frameshift_mutations = False

    seq_len = len(dna_seq)
    i = 0

    while i + 3 <= seq_len:
        codon = dna_seq[i:i+3]
        if '-' in codon:
            protein_seq.append('-')  # Handle gaps as gaps
        elif re.search(r'[^ATCG]', codon):
            protein_seq.append('X')  # Ambiguous nucleotide
        else:
            if codon in stop_codons:
                if to_stop:
                    break
                elif handle_stops == 'remove':
                    pass
                elif handle_stops == 'replace':
                    protein_seq.append('X')
                else:
                    protein_seq.append('*')
            else:
                amino_acid = standard_table.get(codon, 'X')
                protein_seq.append(amino_acid)
        i += 3

    # Handle trailing bases not forming a full codon
    if i < seq_len:
        remaining = dna_seq[i:]
        if '-' in remaining or re.search(r'[^ATCG]', remaining):
            protein_seq.append('-')

    protein_str = ''.join(protein_seq)

    # Detect potential frameshifts
    if ref_protein_length is not None:
        translated_length = len(protein_seq)
        length_difference = abs(translated_length - ref_protein_length)
        tolerance = max(1, int(ref_protein_length * 0.05))  # Allow 5% difference

        frameshift_mutations = length_difference > tolerance

    # Flag internal stop codons (ignore last)
    if '*' in protein_str[:-1]:
        frameshift_mutations = True

    return protein_str, frameshift_mutations


def align_and_handle_deletions(translated_seq, ref_seq):
    """
    Align translated sequence with the reference sequence, handling mismatches and deletions.

    Args:
        translated_seq (str): The translated query protein sequence.
        ref_seq (str): Reference protein sequence.

    Returns:
        aligned_seq_str (str): Aligned sequence with gaps.
    """
    aligned_seq = []
    ref_index = 0
    trans_index = 0

    while ref_index < len(ref_seq) and trans_index < len(translated_seq):
        if translated_seq[trans_index] == ref_seq[ref_index]:
            aligned_seq.append(translated_seq[trans_index])
            trans_index += 1
        elif translated_seq[trans_index] == '-':
            aligned_seq.append('-')
            trans_index += 1
        elif ref_seq[ref_index] == '-':
            ref_index += 1
            continue
        else:
            aligned_seq.append(translated_seq[trans_index])
            trans_index += 1
        ref_index += 1

    # Trim trailing gaps
    aligned_seq_str = ''.join(aligned_seq).rstrip('-')
    return aligned_seq_str


def map_dna_gaps_to_protein_gaps(gap_indices, dna_seq_length):
    """
    Map DNA gap indices to protein sequence gap indices.

    Args:
        gap_indices (list): Indices of gaps in the DNA sequence.
        dna_seq_length (int): Length of the DNA sequence.

    Returns:
        protein_gap_indices (list): Corresponding positions in protein space.
    """
    protein_gap_indices = []
    for gap in gap_indices:
        protein_gap = gap // 3
        if protein_gap < dna_seq_length // 3:
            protein_gap_indices.append(protein_gap)
    return protein_gap_indices


def write_fasta_with_metadata_from_df(df, output_file, reference_length):
    """
    Write protein sequences to a FASTA file with metadata headers.

    Args:
        df (pd.DataFrame): DataFrame with columns ['Filename', 'Protein_Sequence', 'Phenotype', 'Frameshift_Mutation'].
        output_file (str): Path to output FASTA file.
        reference_length (int): Length of the reference protein for annotation.
    """
    with open(output_file, "w") as fasta_file:
        for _, row in df.iterrows():
            filename = row["Filename"]
            sequence = row["Protein_Sequence"]
            phenotype = row["Phenotype"]
            frameshift_flag = row["Frameshift_Mutation"]
            seq_len = row["seq_len"]

            header = f">{filename} | {phenotype} | {seq_len} | Frameshift: {frameshift_flag}"
            fasta_file.write(header + "\n")
            fasta_file.write(sequence + "\n")
