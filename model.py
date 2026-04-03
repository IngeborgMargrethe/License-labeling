# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models

# # ------------------------------------------------
# # Device
# # ------------------------------------------------

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)


# # ------------------------------------------------
# # CNN for CIFAR-10
# # ------------------------------------------------

# class CNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super().__init__()

#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),        # 16x16

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),        # 8x8

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)         # 4x4
#         )

#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128 * 4 * 4, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x


# # ------------------------------------------------
# # Train function
# # ------------------------------------------------

# def train_model(model, dataloader, epochs=5, lr=1e-3):

#     model = model.to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()

#     model.train()

#     for epoch in range(epochs):

#         total_loss = 0

#         for images, labels in dataloader:

#             images = images.to(device)
#             labels = labels.to(device)

#             optimizer.zero_grad()

#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss:.3f}")

        

#     return model


# # ------------------------------------------------
# # Evaluation
# # ------------------------------------------------

# def evaluate_model(model, dataloader):

#     model.eval()

#     correct = 0
#     total = 0

#     with torch.no_grad():

#         for images, labels in dataloader:

#             images = images.to(device)
#             labels = labels.to(device)

#             outputs = model(images)

#             preds = torch.argmax(outputs, dim=1)

#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#     accuracy = correct / total

#     return accuracy


# # ------------------------------------------------
# # Predict probabilities (needed for active learning)
# # ------------------------------------------------

# def predict_probabilities(model, dataloader):

#     model.eval()

#     probs = []

#     with torch.no_grad():

#         for images, _ in dataloader:

#             images = images.to(device)

#             outputs = model(images)

#             p = F.softmax(outputs, dim=1)

#             probs.append(p.cpu())

#     probs = torch.cat(probs)

#     return probs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ------------------------------------------------
# Device
# ------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------------------------------------
# Model
# ------------------------------------------------

class CNN(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, freeze_backbone=True):
        super().__init__()

        # Load ResNet18
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            self.backbone = models.resnet18(weights=weights)
        else:
            self.backbone = models.resnet18(weights=None)

        # Replace final classification layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        # Freeze all backbone layers if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Unfreeze final classification layer
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


# ------------------------------------------------
# Train function
# ------------------------------------------------

def train_model(model, dataloader, epochs=5, lr=1e-3):
    model = model.to(device)

    # Only optimize trainable parameters
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}  Loss: {avg_loss:.4f}")

    return model


# ------------------------------------------------
# Evaluation
# ------------------------------------------------

def evaluate_model(model, dataloader):
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


# ------------------------------------------------
# Predict probabilities
# ------------------------------------------------

def predict_probabilities(model, dataloader):
    model = model.to(device)
    model.eval()

    probs = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)

            outputs = model(images)
            p = F.softmax(outputs, dim=1)

            probs.append(p.cpu())

    probs = torch.cat(probs, dim=0)
    return probs