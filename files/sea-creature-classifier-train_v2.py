import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import json
import torch.optim as optim
import random
from torchvision.transforms import functional as TF
import numpy as np

with open('./dataset/label_counts2.json', 'r') as f:
    label_counts = json.load(f)

total_samples = sum(label_counts.values())
class_weights = {class_name.upper(): total_samples / count for class_name, count in label_counts.items()}

with open('./dataset/labels2.json', 'r') as f:
    label_map = json.load(f)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, augment_prob=0.33):
        self.data = data
        self.transform = transform
        self.augment_prob = augment_prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['path']).convert('RGB')
        label = label_map[item['label'].upper()]
        if random.random() < self.augment_prob:
            image = self.apply_augmentations(image)
        if self.transform:
            image = self.transform(image)
        return image, label

    def apply_augmentations(self, image):
        # Random horizontal flip
        if random.random() > 0.15:
            image = TF.hflip(image)
        # Random vertical flip
        if random.random() > 0.15:
            image = TF.vflip(image)
        # Random rotation
        angle = random.randint(-30, 30)
        if random.random() > 0.15:
            image = TF.rotate(image, angle)
        # Color Jitter
        if random.random() > 0.15:
            image = TF.adjust_brightness(image, random.uniform(0.5, 1.5))
        if random.random() > 0.15:
            image = TF.adjust_contrast(image, random.uniform(0.5, 1.5))
        if random.random() > 0.15:
            image = TF.adjust_saturation(image, random.uniform(0.5, 1.5))
        # Optional: Add random noise
        if random.random() > 0.15:
            numpy_array = np.array(image)
            noise = np.random.normal(0, 0.1, numpy_array.shape)
            numpy_array = np.clip(numpy_array + noise * 255, 0, 255).astype(np.uint8)
            image = Image.fromarray(numpy_array)
        return image

def train_and_test_model(train_data, test_data, label_map, batch_size=32, num_epochs=10, learning_rate=0.001):
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = CustomDataset(train_data, transform=data_transforms, augment_prob=0.1)
    test_dataset = CustomDataset(test_data, transform=data_transforms, augment_prob=0.0)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model = timm.create_model('resnest50d', pretrained=False, num_classes=len(label_map))
    model.to(device)

    model.load_state_dict(torch.load(f'resnest50d_epoch_15.pth', map_location=device))

    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([class_weights[class_name] for class_name in sorted(label_counts.keys())]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'resnest50d_epoch_{epoch+1}.pth')
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                print(f'Test Accuracy: {(correct / total) * 100:.2f}%, total:{total}')

train_data = json.load(open('./dataset/train_labels2.json', 'r'))
test_data = json.load(open('./dataset/test_labels2.json', 'r'))

train_and_test_model(train_data, test_data, label_map, batch_size=64, num_epochs=200, learning_rate=0.0001)
