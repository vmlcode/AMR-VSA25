from utils import get_labels_dataset
from itertools import product
from Bio.SeqUtils import gc_fraction, molecular_weight
import pandas as pd

def get_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


def get_all_kmers_dict():
    bases = ['A', 'C', 'G', 'T']
    kmer_dict = {}
    kmers = [''.join(p) for p in product(bases, repeat=4)]
    for kmer in kmers:
            kmer_dict[kmer] = 0
    return kmer_dict 


def get_kmer_frequencies(sequence, k):
  kmer_list = get_kmers(sequence, k)
  kmer_dict = get_all_kmers_dict()
  for kmer in kmer_list:
        if kmer in kmer_dict:
         kmer_dict[kmer] += 1
  return kmer_dict


def get_gc_content(sequence):
    return gc_fraction(sequence)


def get_sequence_length(sequence):
    return len(sequence)


def get_at_ratio(sequence):
    a_count = sequence.count('A')
    t_count = sequence.count('T')
    return (a_count + t_count) / len(sequence) if len(sequence) > 0 else 0


def get_molecular_weight(sequence):
  return molecular_weight(sequence, seq_type='DNA', monoisotopic=True)

def create_kmer_dataset():
    df = get_labels_dataset()
    new_rows = []

    for i in range(len(df)):
        sequence = df['sequence'][i]
        kmer_frequencies = get_kmer_frequencies(sequence, 4)
        gc_content = get_gc_content(sequence)

        # Start with the original row (excluding 'sequence')
        row_data = df.drop(columns=['sequence']).iloc[i].to_dict()

        # Add k-mer frequencies to it
        row_data.update(kmer_frequencies)
        row_data['gc_content'] = gc_content
        row_data['sequence_length'] = get_sequence_length(sequence)
        row_data['at_ratio'] = get_at_ratio(sequence)
        row_data['molecular_weight'] = get_molecular_weight(sequence)

        new_rows.append(row_data)

    result_df = pd.DataFrame(new_rows)
    result_df.to_csv("../../data/processed/kmer_frecuency_dataset.csv", index=False)

create_kmer_dataset()
