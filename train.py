import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import get_data_loaders
from model import SnoutNet
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def train_model_with_augmentations(apply_flip=False, apply_color_jitter=False, plot_path='training_loss_plot.png', batch_size=64):
    num_classes = 4  # Update based on your dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = SnoutNet(num_classes=num_classes).to(device)
    print(f"\nStarting training with batch size: {batch_size}\n")

    # Define augmentations
    augmentations = []
    if apply_flip:
        augmentations.append(transforms.RandomHorizontalFlip(p=0.5))
    if apply_color_jitter:
        augmentations.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))

    # Load data
    train_loader, val_loader = get_data_loaders(
        'data/images-original/images', 'data/train-labels.txt', 'data/test-labels.txt',
        batch_size=batch_size, augmentations=augmentations
    )

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    loss_values = []
    for epoch in range(200):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for _, images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        loss_values.append(epoch_loss)
        print(f"🔹 Epoch [{epoch + 1}/200] - Loss: {epoch_loss:.4f} - Accuracy: {accuracy:.2f}%")

        scheduler.step()

        # Plot and save training loss
        plt.plot(range(1, epoch + 2), loss_values, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.savefig(plot_path)
        plt.close()

    print("\nTraining complete! Model saved as 'snoutnet_weights.pth'")
    torch.save(model.state_dict(), 'snoutnet_weights.pth')


if __name__ == "__main__":
    apply_flip = input("Flip images? (Y/N): ").strip().lower() == 'y'
    apply_color_jitter = input("Apply color jitter? (Y/N): ").strip().lower() == 'y'
    train_model_with_augmentations(apply_flip=apply_flip, apply_color_jitter=apply_color_jitter, plot_path='training_loss_plot.png', batch_size=64)
