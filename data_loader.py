import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class SurgicalToolsDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        """
        Args:
            img_dir (str): Directory containing images.
            label_file (str): Path to the CSV file with annotations.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = []

        # Read the label file
        with open(label_file, 'r') as file:
            next(file)  # Skip header if present
            for line in file:
                line = line.strip()
                try:
                    img_name, class_label = line.split(',')
                    self.img_labels.append((img_name.strip(), int(class_label.strip())))
                except ValueError:
                    print(f"Skipping line with incorrect format: {line}")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, class_label = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(class_label, dtype=torch.long)

def get_data_loaders(img_dir, label_file, batch_size=32, augmentations=None, validation_split=0.2):
    """
    Returns DataLoader objects for training and validation datasets.
    """
    if augmentations is None:
        augmentations = []

    # Define transforms
    transform = transforms.Compose(
        [transforms.Resize((227, 227))] + augmentations + [transforms.ToTensor()]
    )

    # Create dataset
    dataset = SurgicalToolsDataset(
        img_dir=img_dir, label_file=label_file, transform=transform
    )

    # Split dataset into training and validation
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(validation_split * dataset_size)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    # Create DataLoaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader
