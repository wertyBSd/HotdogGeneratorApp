import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

# === Параметры ===
img_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_paths = {
    'ResNet50': 'hotdog_model_resnet.pt',
    'MobileNetV2': 'hotdog_model_mobilenet.pt',
    'EfficientNetB0': 'hotdog_model_efficientnet.pt'
}

# === Предобработка изображения ===
transform_eval = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === Функция загрузки модели ===
def load_model(model_name, path):
    if model_name == 'ResNet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1), nn.Sigmoid())
    elif model_name == 'MobileNetV2':
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.classifier[1].in_features, 1), nn.Sigmoid())
    elif model_name == 'EfficientNetB0':
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.classifier[1].in_features, 1), nn.Sigmoid())
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# === Загрузка всех моделей ===
models_list = []
model_names = []
for name, path in model_paths.items():
    if os.path.exists(path):
        model = load_model(name, path)
        models_list.append(model)
        model_names.append(name)
    else:
        print(f"[⚠️] Модель не найдена: {path}")

# === Предсказание по одной картинке ===
def predict_image_ensemble(image_path):
    if not os.path.exists(image_path):
        print(f"[❌] Файл не найден: {image_path}")
        return

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform_eval(img).unsqueeze(0).to(device)

    predictions = []
    for i, model in enumerate(models_list):
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.item()
            predictions.append(pred)
            outputPred = 1 - pred
            print(f"[🔍] {model_names[i]} → {outputPred:.4f}")

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
predict_folder_ensemble('dataset_test')