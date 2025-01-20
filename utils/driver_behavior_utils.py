import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

class DriverBehaviorDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.data = []

        for class_name in os.listdir(folder):
            class_path = os.path.join(folder, class_name)
            if os.path.isdir(class_path):
                for file_name in os.listdir(class_path):
                    self.data.append((os.path.join(class_path, file_name), class_name))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        image = Image.open(file_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def load_data(train_folder, test_folder, batch_size=32):
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_dataset = DriverBehaviorDataset(train_folder, transform=transform)
    test_dataset = DriverBehaviorDataset(test_folder, transform=transform)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def calculate_accuracy(model, data_loader, device):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = torch.tensor([int(label) for label in labels]).to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def plot_confusion_matrix(cm, class_names):
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(cm, cmap='coolwarm')
    plt.colorbar(cax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
