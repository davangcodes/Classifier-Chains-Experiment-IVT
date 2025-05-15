"""
==========================================================
🔬 Verb Classification with ResNet18 - Classifier Chain on 'clipper'
==========================================================

🎯 Task: Multi-label verb classification (10 classes)
🧠 Model: ResNet18 (ImageNet pretrained)
⚙️ Loss Function: BCEWithLogitsLoss (multi-label)
🚀 Optimizer: Adam (lr = 0.001)
📊 Evaluation: AP per class + Restricted mAP (Classes 0, 1, 4, 9)
🎯 Classifier Chain: Verb labels conditioned on the presence of 'clipper' (instrument index 4)

🔍 Dataset:
- Format: JSON (base64-encoded images + 'verb_labels' + 'instrument_labels')
- Filter: Only frames where instrument_labels[4] == 1 (i.e., clipper is present)

Author: Davang Sikand
"""

# =================== Imports =================== #
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import json
import base64
from PIL import Image
import numpy as np
from tqdm import tqdm
from io import BytesIO
from sklearn.metrics import average_precision_score

# =================== Dataset =================== #
class VerbDataset(Dataset):
    """
    Dataset class to load surgical images and verb labels.
    Only retains samples where 'clipper' (index 4) is present in instrument_labels.
    """
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as f:
            data = json.load(f)

        print(f"🔍 Total dataset samples (before filtering): {len(data)}")

        self.data = [d for d in data if d["instrument_labels"][4] == 1]
        print(f"✅ Dataset samples (after filtering 'clipper'): {len(self.data)}")

        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image = Image.open(BytesIO(base64.b64decode(item['image'])))
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"🚨 Error loading image at index {idx}: {e}")
            return None
        
        labels = torch.tensor(item['verb_labels'], dtype=torch.float32)
        return image, labels

# =================== Transforms =================== #
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomChoice([
            transforms.RandomVerticalFlip(0.4),
            transforms.RandomHorizontalFlip(0.4),
            transforms.ColorJitter(brightness=0.1, contrast=0.2),
            transforms.RandomRotation(30),
            transforms.RandomAdjustSharpness(sharpness_factor=1.2, p=0.3),
            transforms.RandomAutocontrast(p=0.3)
        ]),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
}

# =================== Class Weights =================== #
def compute_class_weights(dataset):
    """
    Compute inverse frequency class weights (optional).
    """
    label_counts = np.zeros(10)
    for _, labels in dataset:
        label_counts += labels.numpy()

    print(f"✅ Class distribution in dataset: {label_counts}")

    if np.sum(label_counts) == 0:
        print("🚨 WARNING: No positive labels found!")

    total_samples = len(dataset)
    class_weights = total_samples / (label_counts + 1e-6)
    class_weights /= np.max(class_weights)

    return torch.tensor(class_weights, dtype=torch.float32).to(device)

# =================== Device Setup =================== #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================== Load Datasets =================== #
train_dataset = VerbDataset("../instrument_verb_train.json", transform=data_transforms['train'])
val_dataset = VerbDataset("../instrument_verb_val.json", transform=data_transforms['val'])

class_weights = compute_class_weights(train_dataset)

# Optional class distribution print
def compute_class_distribution(dataset, dataset_name="Dataset"):
    label_counts = np.zeros(10)
    for _, labels in dataset:
        label_counts += labels.numpy()
    print(f"✅ Class distribution in {dataset_name}: {label_counts}")

compute_class_distribution(val_dataset, "Validation Dataset")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# =================== Model Definition =================== #
class ResNetVerbClassifier(nn.Module):
    """
    Pretrained ResNet18 modified for 10-label verb classification.
    """
    def __init__(self, num_classes=10):
        super(ResNetVerbClassifier, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

model = ResNetVerbClassifier().to(device)

# =================== Loss & Optimizer =================== #
criterion = nn.BCEWithLogitsLoss()  # You can add pos_weight=class_weights if needed
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =================== Evaluation =================== #
def evaluate_model(model, val_loader):
    """
    Evaluates model on validation set using AP per class and restricted mAP.
    """
    model.eval()
    all_targets, all_preds = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = torch.sigmoid(model(images))  # Convert logits to probabilities

            all_targets.append(labels.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    if np.sum(all_targets) == 0:
        print("🚨 WARNING: No positive class found in validation labels!")

    try:
        AP_per_class = [average_precision_score(all_targets[:, i], all_preds[:, i]) for i in range(10)]
    except ValueError:
        print("🚨 ERROR: Some classes have no positive samples!")
        AP_per_class = [0.0] * 10

    selected_classes = [0, 1, 4, 9]  # Focus on key verbs
    selected_APs = [AP_per_class[i] for i in selected_classes if not np.isnan(AP_per_class[i])]
    restricted_mAP = np.mean(selected_APs) if selected_APs else 0.0

    print("\n📊 AP per Class:")
    for i, ap in enumerate(AP_per_class):
        print(f"✅ AP for Class {i}: {ap:.4f}")
    print(f"\n🔥 Restricted mAP (Class 0, 1, 4, 9): {restricted_mAP:.4f}")

    return AP_per_class, restricted_mAP

# =================== Training =================== #
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    """
    Training loop with validation and checkpointing.
    """
    best_mAP = 0
    best_AP_per_class = None
    best_model_wts = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        AP_per_class, val_mAP = evaluate_model(model, val_loader)

        if val_mAP > best_mAP:
            best_mAP = val_mAP
            best_AP_per_class = AP_per_class
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, "best_resnet_verb.pth")

        print(f"📉 Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Val mAP: {val_mAP:.4f}")

    # Save best AP scores
    with open("best_results.json", "w") as f:
        json.dump({"best_mAP": best_mAP, "AP_per_class": best_AP_per_class}, f, indent=4)

    print("\n✅ Training complete. Best model weights saved.")
    print(f"🔥 Best mAP: {best_mAP:.4f}")
    print("\n📊 Best AP per Class:")
    for i, ap in enumerate(best_AP_per_class):
        print(f"✅ Best AP for Class {i}: {ap:.4f}")

# =================== Run Training =================== #
if __name__ == "__main__":
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)
    print("✅ Finished training with filtered 'clipper' instrument data.")
