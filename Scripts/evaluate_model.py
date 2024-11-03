import argparse
from ultralytics import YOLO
import os
from pathlib import Path

def evaluate_model(model_name):
    # Get the absolute path of the current script
    script_dir = Path(__file__).resolve().parent

    # Construct paths relative to the current script's directory
    model_path = script_dir.parent / 'Models' / model_name
    data_file = script_dir.parent / 'Dataset' /  'data.yaml'

    # Check if the model file exists
    if not model_path.is_file():
        print(f"Error: Model file '{model_path}' not found.")
        return

    # Load the YOLO model
    model = YOLO(str(model_path))

    # Evaluate the model on the validation dataset
    results = model.val(data=str(data_file))  # You can specify other validation parameters if needed

    # Print evaluation metrics (for demonstration purposes)
    print("Evaluation Results:")
    print(results)

    mean_results = results.mean_results()
    # Extracting precision and recall for F1 score calculation
    precision = mean_results[0]
    recall = mean_results[1]

    # Calculating the F1 score using the standard formula
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Displaying the metrics
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"mAP50: {mean_results[2]:.4f}")
    print(f"mAP50-95: {mean_results[3]:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate a YOLO model on the validation dataset.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to evaluate (with extension)')

    args = parser.parse_args()

    # Evaluate the specified model
    evaluate_model(args.model_name)
