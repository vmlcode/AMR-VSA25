import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_model_comparison(analytics_df, output_dir, run_id):
    """
    Plot bar charts comparing accuracy, test loss, and F1-score across models.
    """
    plt.figure(figsize=(12, 5))
    sns.barplot(data=analytics_df, x='model_unique_id', y='accuracy', hue='model_type')
    plt.title('Final Accuracy per Model')
    plt.ylabel('Accuracy')
    plt.xlabel('Model ID')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"accuracy_comparison_{run_id}.png"))
    plt.close()

    plt.figure(figsize=(12, 5))
    sns.barplot(data=analytics_df, x='model_unique_id', y='loss_test', hue='model_type')
    plt.title('Test Loss per Model')
    plt.ylabel('Test Loss')
    plt.xlabel('Model ID')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"loss_comparison_{run_id}.png"))
    plt.close()

    if "f1_score" in analytics_df.columns:
        plt.figure(figsize=(12, 5))
        sns.barplot(data=analytics_df, x='model_unique_id', y='f1_score', hue='model_type')
        plt.title('F1 Score per Model')
        plt.ylabel('F1 Score')
        plt.xlabel('Model ID')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"f1_score_comparison_{run_id}.png"))
        plt.close()

def plot_training_history(histories, model_type, encoding, run_id, output_dir):
    """
    Plot training and validation loss and accuracy for a list of model histories.
    """
    for idx, history in enumerate(histories):
        history_dict = history
        epochs = range(1, len(history_dict['loss']) + 1)

        # Try to get model_unique_id if present
        model_id = history_dict.get('model_unique_id', f"{model_type}_{idx+1}")

        plt.figure(figsize=(12, 5))
        plt.suptitle(f'{model_type} Model {model_id} - {encoding} - Run {run_id}', fontsize=14)

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history_dict['loss'], label=f'Training Loss ({model_id})')
        plt.plot(epochs, history_dict['val_loss'], label=f'Validation Loss ({model_id})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history_dict['accuracy'], label=f'Training Accuracy ({model_id})')
        plt.plot(epochs, history_dict['val_accuracy'], label=f'Validation Accuracy ({model_id})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{model_type.lower()}_{encoding}_model_{model_id}_run_{run_id}.png")
        plt.savefig(plot_path)
        plt.close()

