import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# === Параметры ===
data_dir = 'dataset'
img_size = 224
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Предобработка ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === Загрузка данных ===
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

# === Обучение модели ===
def build_and_train_model(base_model_fn, model_name):
    print(f"\n📦 Обучение модели {model_name}...")
    base_model = base_model_fn(weights='DEFAULT')

    # Универсальная замена последнего слоя
    if hasattr(base_model, 'fc'):  # ResNet
        in_features = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )
    elif hasattr(base_model, 'classifier'):  # MobileNetV2, EfficientNetB0
        if isinstance(base_model.classifier, nn.Sequential):
            in_features = base_model.classifier[-1].in_features
        else:
            in_features = base_model.classifier.in_features
        base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )
    else:
        raise ValueError(f"Неизвестная архитектура модели: {model_name}")

    model = base_model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_acc, val_acc = [], []
    train_loss, val_loss = [], []

    for epoch in range(5):
        model.train()
        correct, total, running_loss = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc.append(correct / total)
        train_loss.append(running_loss / len(train_loader))

        # Валидация
        model.eval()
        correct, total, running_loss = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc.append(correct / total)
        val_loss.append(running_loss / len(val_loader))

        print(f"Epoch {epoch+1}: Train Acc={train_acc[-1]:.4f}, Val Acc={val_acc[-1]:.4f}")

    # Сохранение модели
    torch.save(model.state_dict(), f"{model_name}.pt")
    print(f"✅ Сохранено: {model_name}.pt")

    # Графики
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Train')
    plt.plot(val_acc, label='Val')
    plt.title(f'{model_name} Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label='Val')
    plt.title(f'{model_name} Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{model_name}_training.png')
    plt.close()
    print(f"📈 Сохранён график: {model_name}_training.png\n")

    return model

# === Обучение всех моделей ===
models_raw = [
    (build_and_train_model(models.resnet50, 'hotdog_model_resnet'), 'ResNet50'),
    (build_and_train_model(models.mobilenet_v2, 'hotdog_model_mobilenet'), 'MobileNetV2'),
    (build_and_train_model(models.efficientnet_b0, 'hotdog_model_efficientnet'), 'EfficientNetB0')
]

models_list = [m for m, name in models_raw if m is not None]
model_names = [name for m, name in models_raw if m is not None]

# === Предсказание по одной картинке ===
def predict_image_ensemble(image_path):
    if not os.path.exists(image_path):
        print(f"[❌] Файл не найден: {image_path}")
        return

    img = Image.open(image_path).convert("RGB")
    transform_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img_tensor = transform_eval(img).unsqueeze(0).to(device)

    predictions = []
    for i, model in enumerate(models_list):
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.item()
            predictions.append(pred)
            print(f"[🔍] {model_names[i]} → {pred:.4f}")

    average = np.mean(predictions)
    label = "HOTDOG 🌭" if average < 0.1 else "NOT HOTDOG ❌"
    confidence = (1 - average) * 100
    print(f"[🎯] Ensemble → {label} | Confidence: {confidence:.2f}%\n")

# === Предсказание по папке ===
def predict_folder_ensemble(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            predict_image_ensemble(image_path)

# === Запуск ===
print("🔍 Классы:", train_dataset.class_to_idx)
predict_folder_ensemble('dataset_test')