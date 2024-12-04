from ultralytics import YOLO

# Train the YOLO model with the specified configuration
def train_model(model_file):
    try:
        print(f"\n[TRAINING MODE]")
        print(f"Loading model: {model_file}...")
        model = YOLO(model_file)  # Load the specified YOLO model
        print("Model loaded successfully.")

        # Train the model with dataset and parameters
        model.train(
            data="surgical_tools.yaml",  # Path to the dataset configuration file
            epochs=200,                 # Number of training epochs
            imgsz=640,                  # Image size for training
            save=True,                  # Save training checkpoints
            augment=True                # Enable data augmentation
        )
        print("\nTraining complete! Results saved in the default 'runs' directory.")

        # Validate the trained model
        print("\nStarting validation...")
        model.val(data="surgical_tools.yaml", save=True, plots=True)
        print("\nValidation complete! Results saved in the default 'runs' directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Prompt user to select a YOLO model
    print("Select the YOLO model:")
    print("1. YOLO v11\n2. YOLO v8")
    choice = input("\nEnter your choice (1 or 2): ").strip()

    # Assign model file based on user selection
    model_file = "yolo11n.pt" if choice == "1" else "yolov8n.pt" if choice == "2" else None

    # Train the selected model or exit on invalid choice
    if model_file:
        train_model(model_file)
    else:
        print("Invalid choice. Exiting.")
