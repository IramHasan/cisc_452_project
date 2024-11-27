import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def visualize_predictions(test_loader, model, img_dir, num_samples=27):
    """
    Visualize predictions for a classification model.
    
    Args:
        test_loader: DataLoader for the test dataset.
        model: Trained classification model.
        img_dir: Directory containing test images.
        num_samples: Number of samples to visualize.
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_images = []

    for _, (img_names, test_images, true_labels) in enumerate(test_loader):
        test_images, true_labels = test_images.to(device), true_labels.to(device)

        with torch.no_grad():
            predicted_probs = model(test_images)
            predicted_labels = torch.argmax(predicted_probs, dim=1)

        for i in range(test_images.size(0)):
            image = test_images[i].permute(1, 2, 0).cpu().numpy()
            image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize for display

            true_class = true_labels[i].item()
            predicted_class = predicted_labels[i].item()
            img_name = img_names[i]

            all_images.append((image, true_class, predicted_class, img_name))

    # Select random samples if more images than num_samples
    if len(all_images) > num_samples:
        all_images = np.random.choice(all_images, num_samples, replace=False)

    # Display images
    for fig_idx in range((num_samples + 8) // 9):  # 9 images per figure
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()

        for i, (image, true_class, predicted_class, img_name) in enumerate(all_images[fig_idx * 9: (fig_idx + 1) * 9]):
            ax = axes[i]
            ax.imshow(image)
            ax.set_title(f"{img_name}\nTrue: {true_class}, Predicted: {predicted_class}",
                         color='green' if true_class == predicted_class else 'red')
            ax.axis('off')

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from data_loader import get_data_loaders
    from model import SnoutNet

    img_dir = 'data/images-original/images'
    batch_size = 64

    num_classes = 4  # Update based on the dataset
    model = SnoutNet(num_classes=num_classes).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(torch.load('snoutnet_weights.pth'))

    test_loader = get_data_loaders(img_dir, 'data/train-labels.txt', 'data/test-labels.txt', batch_size)[1]
    visualize_predictions(test_loader, model, img_dir, num_samples=27)
