
from preprocessing.labels_dataset import populate_amr_fasta_dataset
from preprocessing.utils import get_dataset
from preprocessing.kmer_and_features_dataset import create_kmer_dataset
# from preprocessing.fcgr_dataset import create_fcgr_dataset

import os
import json

def run_ml_pipeline():
    """
    Run the entire machine learning pipeline.    
    """
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    # Define paths relative to project root
    data_dir = os.path.join(project_root, 'Examples')
    labels_dir = os.path.join(project_root, 'data', 'labels.csv')
    kmer_features_dir = os.path.join(project_root, 'data', 'processed' ,'kmer_features.csv')
    # fcgr_dataset_dir = os.path.join(project_root, 'data', 'processed', 'fcgr_dataset.csv')



    # Run the data population step with absolute paths
    populate_amr_fasta_dataset(data_dir, labels_dir)
    
    # Load the labels dataset
    labels_df = get_dataset(labels_dir)
    print(labels_df.head())  # Display the first few rows of the labels dataset

    # Create the kmer and features dataset
    create_kmer_dataset(kmer_features_dir, labels_df)
    kmer_features_dataset = get_dataset(kmer_features_dir)
    print(kmer_features_dataset.head())  # Display the first few rows of the kmer features dataset

    # Create the fcgr dataset
    # create_fcgr_dataset(fcgr_dataset_dir, labels_df)
    # fcgr_dataset = get_dataset(fcgr_dataset_dir)
    # print(fcgr_dataset.head())  

    # divide the datasets into train and test sets
    train_set_kmer = kmer_features_dataset.sample(frac=0.8, random_state=42)
    test_set_kmer = kmer_features_dataset.drop(train_set_kmer.index)

    # train_set_fcgr = fcgr_dataset.sample(frac=0.8, random_state=42)
    # test_set_fcgr = fcgr_dataset.drop(train_set_fcgr.index)

    

    # create the models
    with open(os.path.join(project_root, 'model_config.json'), 'r') as f:
        model_builder = json.load(f)

    for i in range(len(model_builder["models"])):
        print(model_builder["models"][i])

    










def main():
    """
    Main entry point for the pipeline.
    """
    run_ml_pipeline()

if __name__ == "__main__":
    main()
