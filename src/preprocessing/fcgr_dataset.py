import numpy as np
import pandas as pd
from utils import get_labels_dataset

def base_to_vertex(base):
    mapping = {
        'A': np.array([1,0]),
        'G': np.array([0,1]),
        'C': np.array([1,1]),
        'T': np.array([0,0]),
    }
    return np.array(mapping.get(base, np.array([0,0])))

def chaos_game_point(kmer):
    point= np.array([0.5, 0.5])
    for base in kmer:
        vertex = base_to_vertex(base)
        if vertex is None:
            return None
        point = point + vertex
        point = point / 2
    return point

def kmer_to_index(point, resolution):
    x = int(point[0] * resolution)
    y = int(point[1] * resolution)
    return x, y

def generate_fcgr(sequence, k, resolution):
    fcgr = np.zeros((resolution, resolution))
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        point = chaos_game_point(kmer)
        if point is not None:
            x, y = kmer_to_index(point, resolution)
            fcgr[x,y] += 1
    return fcgr

def create_amr_fasta_fcgr_dataset():
    df = get_labels_dataset()
    new_rows = []
    for i in range(len(df)):
        sequence = df['sequence'][i]
        fcgr = generate_fcgr(sequence, 6, 64)
        row_data = df.drop(columns=['sequence']).iloc[i].to_dict()
        row_data['fcgr'] = fcgr.flatten().tolist()
        new_rows.append(row_data)
    result_df = pd.DataFrame(new_rows)
    result_df.to_csv("../../data/processed/fcgr_dataset.csv", index=False)

create_amr_fasta_fcgr_dataset()
