import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinter import filedialog

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

# === GUI ===
class HotdogApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hotdog Classifier 🌭")
        self.root.geometry("600x500")
        self.image_path = None

        self.label = tk.Label(root, text="Выберите изображение для классификации", font=("Arial", 14))
        self.label.pack(pady=10)

        self.btn_select = tk.Button(root, text="📁 Выбрать изображение", command=self.select_image)
        self.btn_select.pack(pady=5)

        self.canvas = tk.Canvas(root, width=224, height=224)
        self.canvas.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 16), fg="blue")
        self.result_label.pack(pady=10)

        self.model_results = tk.Text(root, height=6, width=60)
        self.model_results.pack(pady=5)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image_path = file_path
            self.display_image()
            self.predict_image()

    def display_image(self):
        img = Image.open(self.image_path).convert("RGB")
        img = img.resize((224, 224))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    def predict_image(self):
        img = Image.open(self.image_path).convert("RGB")
        img_tensor = transform_eval(img).unsqueeze(0).to(device)

        predictions = []
        self.model_results.delete("1.0", tk.END)

        for i, model in enumerate(models_list):
            with torch.no_grad():
                output = model(img_tensor)
                pred = output.item()
                predictions.append(pred)
                outputPred = 1 - pred
                self.model_results.insert(tk.END, f"{model_names[i]} → {outputPred:.4f}\n")

        average = np.mean(predictions)
        label = "🌭 HOTDOG" if average < 0.3 else "❌ NOT HOTDOG"
        confidence = (1 - average) * 100 if average < 0.3 else average * 100
        self.result_label.config(text=f"{label} | Уверенность: {confidence:.2f}%")

# === Запуск приложения ===
root = tk.Tk()
app = HotdogApp(root)
root.mainloop()