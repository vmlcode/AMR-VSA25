from preprocessing.labels_dataset import populate_amr_fasta_dataset
from preprocessing.utils import get_dataset
from preprocessing.kmer_and_features_dataset import create_kmer_dataset
from preprocessing.fcgr_dataset import create_fcgr_dataset

from models.dnn_model import build_dnn_model
from models.cnn_model import build_cnn_model

from pipeline.utils import generate_model_name

from visualization.plots import plot_training_history, plot_model_comparison, plot_training_all_history, plot_training_history_one_image

from tensorflow.keras.callbacks import EarlyStopping # type: ignore

from time import time

import os
import json
import random
import ast
import numpy as np
import pandas as pd

def run_ml_pipeline():
    """
    Run the entire machine learning pipeline.    
    """

    project_name = "VSA-25"
    run_id = str(time())

    analitics_df = pd.DataFrame(columns=["project_name", "run_id", "model_type", "encoding", "model_unique_id", "epochs", "tag", "accuracy", "loss_training", "loss_test", "f1_score", "history"])

    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    # Define paths relative to project root
    data_dir = os.path.join(project_root, 'data', 'raw')
    labels_dir = os.path.join(project_root, 'data', 'labels.csv')
    kmer_features_dir = os.path.join(project_root, 'data', 'processed' ,'kmer_features.csv')
    fcgr_dataset_dir = os.path.join(project_root, 'data', 'processed', 'fcgr_dataset.csv')

    # Run the data population step with absolute paths
    populate_amr_fasta_dataset(data_dir, labels_dir)
    
    # Load the labels dataset
    labels_df = get_dataset(labels_dir)
    print(labels_df.head())  # Display the first few rows of the labels dataset

    # Create the kmer and features dataset
    if os.path.exists(kmer_features_dir):
        print("Kmer features dataset already exists. Loading from file.")
        kmer_features_dataset = get_dataset(kmer_features_dir)
    else:
        create_kmer_dataset(kmer_features_dir, labels_df)
        kmer_features_dataset = get_dataset(kmer_features_dir)
        print(kmer_features_dataset.head())  # Display the first few rows of the kmer features dataset

    #check if fcgr dataset already exists
    if os.path.exists(fcgr_dataset_dir):
        print("FCGR dataset already exists. Loading from file.")
        fcgr_dataset = get_dataset(fcgr_dataset_dir)
    else:
        # Create the fcgr dataset
        create_fcgr_dataset(fcgr_dataset_dir, labels_df)
        fcgr_dataset = get_dataset(fcgr_dataset_dir)
        print(fcgr_dataset.head())  

    for i in range(len(fcgr_dataset)):
        arr = np.array(ast.literal_eval(fcgr_dataset["fcgr"][i]))
        fcgr_dataset.at[i, "fcgr"] = arr.reshape(64,64)
    

    # divide the datasets into train and test sets
    train_set_kmer = kmer_features_dataset.sample(frac=0.6, random_state=46)
    test_set_kmer = kmer_features_dataset.drop(train_set_kmer.index)

    x_train_kmer = train_set_kmer.drop(columns=['sample_id', 'label (not resistant[0]/resistant[1] to Trimethoprim)'])
    y_train_kmer = train_set_kmer['label (not resistant[0]/resistant[1] to Trimethoprim)'].values.reshape(-1, 1)
    x_test_kmer = test_set_kmer.drop(columns=['sample_id', 'label (not resistant[0]/resistant[1] to Trimethoprim)'])
    y_test_kmer = test_set_kmer['label (not resistant[0]/resistant[1] to Trimethoprim)'].values.reshape(-1, 1)
    print("Train set kmer features shape:", x_train_kmer.shape)
    print("Test set kmer features shape:", x_test_kmer.shape)
    print("Train set kmer labels shape:", y_train_kmer.shape)
    print("Test set kmer labels shape:", y_test_kmer.shape)
    
    
    train_set_fcgr = fcgr_dataset.sample(frac=0.6, random_state=46)
    test_set_fcgr = fcgr_dataset.drop(train_set_fcgr.index)

    # Stack fcgr arrays and add channel dimension
    x_train_fcgr = np.stack(train_set_fcgr["fcgr"].values).astype(np.float32)
    x_train_fcgr = x_train_fcgr[..., np.newaxis]  # shape: (num_samples, 64, 64, 1)
    y_train_fcgr = train_set_fcgr['label (not resistant[0]/resistant[1] to Trimethoprim)'].values.reshape(-1, 1)

    x_test_fcgr = np.stack(test_set_fcgr["fcgr"].values).astype(np.float32)
    x_test_fcgr = x_test_fcgr[..., np.newaxis]  # shape: (num_samples, 64, 64, 1)
    y_test_fcgr = test_set_fcgr['label (not resistant[0]/resistant[1] to Trimethoprim)'].values.reshape(-1, 1)

    print("Train set fcgr features shape:", x_train_fcgr.shape)
    print("Test set fcgr features shape:", x_test_fcgr.shape)
    print("Train set fcgr labels shape:", y_train_fcgr.shape)
    print("Test set fcgr labels shape:", y_test_fcgr.shape)
    

    # create the models
    with open(os.path.join(project_root, 'model_config.json'), 'r') as f:
        model_builder = json.load(f)

    dnn_models = []
    cnn_models = []

    for i in range(len(model_builder["models"])):
        print(model_builder["models"][i])
        if model_builder["models"][i]["type"] == "DNN":
            model = build_dnn_model(
                name=f"{model_builder['models'][i]['id']}_{random.randint(1000, 9999)}",
                hidden_layers=model_builder["models"][i]["hidden_layers"],
                optimizer=model_builder["models"][i]["optimizer"],
                loss=model_builder["models"][i]["loss"],
                metrics=model_builder["models"][i]["metrics"],
                regularization=model_builder["models"][i]["regularization"]
            )
            model.summary()
            dnn_models.append(model)
        if model_builder["models"][i]["type"] == "CNN":
            model = build_cnn_model(
                name=f"{model_builder['models'][i]['id']}_{random.randint(1000, 9999)}",
                hidden_layers=model_builder["models"][i]["hidden_layers"],
                optimizer=model_builder["models"][i]["optimizer"],
                loss=model_builder["models"][i]["loss"],
                metrics=model_builder["models"][i]["metrics"],
                regularization=model_builder["models"][i]["regularization"],
            )
            model.summary()
            cnn_models.append(model)
    
    print(cnn_models, dnn_models)

    dnn_history = []
    cnn_history = []

    # Train the models
    early_stop = EarlyStopping(
        monitor='val_loss',      
        patience=5,              
        verbose=1,              
        restore_best_weights=True  
    )

    for i in range(len(dnn_models)):
        for epochs in model_builder["test_settings"]["epochs"]:
            model_name = generate_model_name(project=project_name, model_type="DNN", encoding="KMER", model_unique_id=dnn_models[i].name, epochs=epochs, tag="RESEARCH", run_id=run_id)
            print(f"Training {dnn_models[i].name} for {epochs} epochs")
            model = dnn_models[i].fit(
                x_train_kmer,
                y_train_kmer,
                epochs=epochs,
                validation_data=(x_test_kmer, y_test_kmer),
                callbacks=[early_stop],
            )
            dnn_models[i].save(os.path.join(project_root, "exports", "models", f"{model_name}.keras"))
            # Add model_unique_id to history
            history = model.history
            history['model_unique_id'] = dnn_models[i].name
            dnn_history.append(history)
            analitics_df = pd.concat([analitics_df, pd.DataFrame([{
                "project_name": project_name,
                "run_id": run_id,
                "model_type": "DNN",
                "encoding": "KMER",
                "model_unique_id": f"{cnn_models[i].name}_{epochs}",
                "epochs": epochs,
                "tag": "RESEARCH",
                "accuracy": model.history['accuracy'][-1],
                "loss_training": model.history['loss'][-1],
                "loss_test": model.history['val_loss'][-1],
                "f1_score": model.history['f1_score'][-1],
                "history": model.history
            }])], ignore_index=True)
    
    for i in range(len(cnn_models)):
        for epochs in model_builder["test_settings"]["epochs"]:
            model_name = generate_model_name(project=project_name, model_type="CNN", encoding="FCGR", model_unique_id=cnn_models[i].name, epochs=epochs, tag="RESEARCH", run_id=run_id)
            print(f"Training {cnn_models[i].name} for {epochs} epochs")
            trained_model = cnn_models[i].fit(
                x_train_fcgr,
                y_train_fcgr,
                epochs=epochs,
                validation_data=(x_test_fcgr, y_test_fcgr),
                callbacks=[early_stop],
            )
            cnn_models[i].save(os.path.join(project_root, "exports", "models", f"{model_name}.keras"))
            # Add model_unique_id to history
            history = trained_model.history
            history['model_unique_id'] = cnn_models[i].name
            cnn_history.append(history)
            analitics_df = pd.concat([analitics_df, pd.DataFrame([{
                "project_name": project_name,
                "run_id": run_id,
                "model_type": "CNN",
                "encoding": "FCGR",
                "model_unique_id": f"{cnn_models[i].name}_{epochs}",
                "epochs": epochs,
                "tag": "RESEARCH",
                "accuracy": trained_model.history['accuracy'][-1],
                "loss_training": trained_model.history['loss'][-1],
                "loss_test": trained_model.history['val_loss'][-1],
                "f1_score": trained_model.history['f1_score'][-1],
                "history": trained_model.history
            }])], ignore_index=True)
            print(trained_model.history["accuracy"][-1])
        
    # Save the analytics DataFrame to a CSV file
    analitics_df.to_csv(os.path.join(project_root, "exports", "reports", f"analytics_{run_id}.csv"), index=False)

    # create evaluation report
    # Define output directory for plots
    plot_output_dir = os.path.join(project_root, "exports", "reports")
    os.makedirs(plot_output_dir, exist_ok=True)

    # Plot training histories
    plot_training_history(dnn_history, "DNN", "KMER", run_id, plot_output_dir)
    plot_training_history(cnn_history, "CNN", "FCGR", run_id, plot_output_dir)
    plot_training_all_history(dnn_history, "DNN", "KMER", run_id, plot_output_dir)
    plot_training_all_history(cnn_history, "CNN", "FCGR", run_id, plot_output_dir)
    plot_training_all_history(dnn_history + cnn_history, "ALL DNN", "ALL CNN", run_id, plot_output_dir)
    # Plot training history for one image
    plot_training_history_one_image(dnn_history, "DNN", "KMER", run_id, plot_output_dir)
    plot_training_history_one_image(cnn_history, "CNN", "FCGR", run_id, plot_output_dir)
    

    # Plot model comparisons
    plot_model_comparison(analitics_df, plot_output_dir, run_id)
            


def main():
    """
    Main entry point for the pipeline.
    """
    run_ml_pipeline()

if __name__ == "__main__":
    main()
