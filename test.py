import os
from ultralytics import YOLO

# Validate the model on the test split defined in the dataset YAML file
def validate_model(model_path):
    try:
        print("\n[VALIDATION MODE]")
        model = YOLO(model_path)
        print("Model loaded successfully.")
        # Perform validation and save results
        results = model.val(data='surgical_tools.yaml', plots=True, save=True, split="test", batch=32)
        print(f"Validation complete! Results saved to: {results.save_dir}")
    except Exception as e:
        print(f"Error during validation: {e}")

# Predict on test images using the trained model
def predict_images(model_path):
    try:
        print("\n[PREDICTION MODE]")
        model = YOLO(model_path)
        print("Model loaded successfully.")

        # Define default paths for test images and output predictions
        test_dir = os.path.join(os.getcwd(), "test/images")
        save_dir = os.path.join(os.getcwd(), "runs/detect")

        # Ensure the test directory exists and contains valid image files
        if not os.path.exists(test_dir):
            print(f"Error: Test directory '{test_dir}' does not exist.")
            return
        image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"Error: No valid image files found in '{test_dir}'.")
            return

        print(f"Found {len(image_files)} image(s) in the test directory.")

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        print(f"Results will be saved in: {save_dir}")

        # Predict on each image and save results
        for image_name in image_files:
            image_path = os.path.join(test_dir, image_name)
            print(f"Processing image: {image_path}")
            model.predict(image_path, save=True, save_dir=save_dir)

        print(f"Predictions complete! Check results in: {save_dir}")
    except Exception as e:
        print(f"Error during predictions: {e}")

if __name__ == "__main__":
    print("Choose an option:\n1. Validate the model.\n2. Predict on test images.")
    choice = input("Enter your choice (1 or 2): ").strip()

    # Request model path and validate it
    model_path = input("Enter the path to the model (.pt file): ").strip()
    if not os.path.isfile(model_path):
        print(f"Error: Model file '{model_path}' not found. Exiting.")
        exit()

    # Execute the selected operation
    if choice == "1":
        validate_model(model_path)
    elif choice == "2":
        predict_images(model_path)
    else:
        print("Invalid choice. Exiting.")
