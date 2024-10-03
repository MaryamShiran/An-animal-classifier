# Import necessary libraries
import torch
from torchvision import datasets, transforms

def get_data_loader():
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to a fixed size
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize images
    ])

    dataset = datasets.ImageFolder('Data/Data/Animals/train', transform = transform)

    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size  # Remaining 20% for testing
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Split the dataset into training and testing sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader
