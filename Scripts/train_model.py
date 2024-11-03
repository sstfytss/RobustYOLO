import argparse
from ultralytics import YOLO
import torch
import os
from pathlib import Path

def train_yolo(yolo_version, epochs, model_name):

    # Get the absolute path of the current script
    script_dir = Path(__file__).resolve().parent

    # Construct paths relative to the current script's directory
    data_file_path = script_dir.parent / 'Dataset' /  'data.yaml'

    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")

    # Load the YOLO model
    model = YOLO(yolo_version)  # Load the specified YOLO model

    # Train the model
    model.train(data=data_file_path, epochs=epochs, imgsz=416, device=device)  # Adjust image size as needed

    # Create the output directory if it doesn't exist
    output_dir = '../Models'

    # Save the model
    model.export(format='pt', name=os.path.join(output_dir, model_name))
    print(f"Model saved as {os.path.join(output_dir, model_name)}.pt")

    yolo_file_path = script_dir / 'yolov8l.pt'
    if yolo_file_path.exists():
        yolo_file_path.unlink()
        
    print(f"Deleted {yolo_file_path}")


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train YOLO model and save the trained weights.')
    parser.add_argument('--version', type=str, required=True, help='YOLO version to use (e.g., yolov8s.pt)')
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--model_name', type=str, required=True, help='Name for the output model (without extension)')

    args = parser.parse_args()

    # Train YOLO model with specified parameters
    train_yolo(args.version, args.epochs, args.model_name)
