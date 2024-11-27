import torch
import torch.nn as nn

class SnoutNet(nn.Module):
    def __init__(self, num_classes=4):
        super(SnoutNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(8 * 8 * 256, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_classes)  # Output layer for classification

    def forward(self, x):
        # Convolutional layers with ReLU activations
        x = self.pool1(nn.ReLU()(self.conv1(x)))
        x = self.pool2(nn.ReLU()(self.conv2(x)))
        x = self.pool3(nn.ReLU()(self.conv3(x)))

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)  # No activation here (handled by loss function in training)

        return x


if __name__ == "__main__":
    num_classes = 4  # Example: Surgical tools classification with 4 categories
    model = SnoutNet(num_classes=num_classes)

    # Dummy input for testing
    dummy_input = torch.randn(1, 3, 227, 227)  # Batch size 1, 3 channels, 227x227 image
    output = model(dummy_input)

    print(f"Output shape: {output.shape}")
    print(f"Predicted class logits: {output}")
