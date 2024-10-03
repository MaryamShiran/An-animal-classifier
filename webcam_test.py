import torch
import torch.nn as nn
import torchvision.transforms as transforms
class CNNModel(nn.Module):
    def __init__(self, num_classes=8):
        super(CNNModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
from PIL import Image
import cv2


# Define the transform for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the trained model
model = CNNModel()
model.load_state_dict(torch.load("trained_model_TrAc95_ValAc99.pth"))
model.eval()

# Function to capture webcam frame and perform prediction
def predict_webcam_frame():
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam")
        return
    # defining the dictionary of classes
    classes = {0 : "bird",
               1 : "cat",
               2 : "cow",
               3 : "dog",
               4 : "elephant",
               5 : "giraffe",
               6 : "horse",
               7 : "zebra"
               }
    # Infinite loop to continuously capture frames
    while True:
        # Read webcam frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Convert frame to PIL Image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Preprocess the image
        image_tensor = transform(frame_pil).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)

        # Get the predicted class label
        predicted_label = predicted.item()

        
        # Display the predicted label on the frame
        cv2.putText(frame, f"Prediction: {classes[predicted_label]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Webcam", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start capturing webcam frames and perform predictions
predict_webcam_frame()
