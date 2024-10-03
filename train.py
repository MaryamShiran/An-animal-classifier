from data_loader import get_data_loader
from model import CNNModel
import torch
import torch.nn as nn


# getting the data_loader
train_loader, test_loader = get_data_loader()
print("data loaded.")
# Initialize the model
model = CNNModel(num_classes=8)

# Setting the hyperparameters
learning_rate = 0.001
num_epochs = 10

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Move the model to the GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("model is being trained on gpu")
else:
    device = torch.device("cpu")
    print("model is being trained on cpu")
model.to(device)

# Training the model
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Move images and labels to the GPU if available
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss for every epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        # Move images and labels to the GPU if available
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100}%")
