# train.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
import matplotlib.pyplot as plt

# Metrics
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ===============================
# 1. Hyperparameters and Data Setup
# ===============================
BATCH_SIZE = 8
NUM_CLASSES = 4
EPOCHS = 25
LEARNING_RATE = 1e-4
DATA_DIR = './data'  # Make sure this path is correct

# 2. Data pre-processing + augmentation for train, standardization for val/test
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ===============================
# 3. Training and Evaluation Functions
# ===============================

def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch and return avg loss & accuracy"""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(loader, desc='Training', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def eval_epoch(model, loader, criterion, device):
    """Evaluate for one epoch and return avg loss & accuracy"""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

def collect_preds_labels(model, loader, device):
    """Collect predictions and true labels for metrics"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1).detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return all_preds, all_labels

# ===============================
# 4. Main Training Loop With Detailed Logging and Plotting
# ===============================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loaders for train, val, and test
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model: EfficientNet-V2-L with new classifier head
    model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    # For plotting and research reporting
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    best_acc = 0.0
    print('Training started...')
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        print(f'Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}')
        print(f'Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}')

        # Store for graphing
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Save the best model checkpoint
        if val_acc > best_acc:
            torch.save(model.state_dict(), 'best_soil_model.pth')
            best_acc = val_acc

    # ========== Plotting Results ==========
    epochs = range(1, EPOCHS+1)
    plt.figure(figsize=(14, 6))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ========== Test Evaluation ==========
    model.load_state_dict(torch.load('best_soil_model.pth', map_location=device))
    model.to(device)
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')

    # ========== Detailed Metrics on Test ==========
    class_names = test_dataset.classes  # folder names as class labels

    y_pred, y_true = collect_preds_labels(model, test_loader, device)

    # Text report (per-class Precision/Recall/F1 + macro/weighted + accuracy)
    print("\nClassification Report (Test):")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Extract numeric summaries if needed for logs
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=True
    )
    overall_acc = report_dict['accuracy']
    macro_p = report_dict['macro avg']['precision']
    macro_r = report_dict['macro avg']['recall']
    macro_f1 = report_dict['macro avg']['f1-score']
    weighted_p = report_dict['weighted avg']['precision']
    weighted_r = report_dict['weighted avg']['recall']
    weighted_f1 = report_dict['weighted avg']['f1-score']
    print(f"\nOverall Accuracy: {overall_acc:.4f}")
    print(f"Macro  P/R/F1: {macro_p:.4f} / {macro_r:.4f} / {macro_f1:.4f}")
    print(f"Weighted P/R/F1: {weighted_p:.4f} / {weighted_r:.4f} / {weighted_f1:.4f}")

    # Confusion matrix (4x4)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Test')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()
