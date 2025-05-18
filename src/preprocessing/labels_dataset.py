import pandas as pd
import os

from Bio import SeqIO

def extract_sequences_from_directory(directory):
    sequences = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        for record in SeqIO.parse(filepath, "fasta"):
            seq = str(record.seq)
            if len(seq) > 0:
                sequences.append([seq, record.id])
    return sequences

def populate_amr_fasta_dataset(dir_path, output_path):
    df = pd.DataFrame(columns=['sample_id', 'sequence', 'label (not resistant[0]/resistant[1] to Trimethoprim)'])
    for directory in os.listdir(dir_path):
        if directory == '0':
            for seq in extract_sequences_from_directory(os.path.join(dir_path, directory)):
                df.loc[len(df)] = [seq[1], seq[0], 0]
        if directory == '1':
            for seq in extract_sequences_from_directory(os.path.join(dir_path, directory)):
                df.loc[len(df)] = [seq[1], seq[0], 1]
    df.to_csv(output_path, index=False)    



