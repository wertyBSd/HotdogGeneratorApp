import os
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

# === Параметры ===
latent_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "generator_hotdog_final.pt"

# === Класс генератора ===

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256 * 8 * 8),
            nn.Unflatten(1, (256, 8, 8)),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


# === Загрузка модели ===
generator = Generator().to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()

# === GUI ===
class HotdogGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hotdog Generator 🌭")
        self.root.geometry("400x400")
        self.image = None

        self.label = tk.Label(root, text="Нажмите кнопку для генерации хотдога", font=("Arial", 14))
        self.label.pack(pady=10)

        self.btn_generate = tk.Button(root, text="Сгенерировать хотдог 🌭", command=self.generate_hotdog)
        self.btn_generate.pack(pady=10)

        self.spinner = ttk.Label(root, text="", font=("Arial", 12))
        self.spinner.pack(pady=5)

        self.canvas = tk.Canvas(root, width=224, height=224)
        self.canvas.pack(pady=10)

    def generate_hotdog(self):
        self.spinner.config(text="⏳ Генерация...")
        self.root.update()

        with torch.no_grad():
            z = torch.randn(1, latent_dim).to(device)
            gen_img = generator(z).detach().cpu()
            gen_img = (gen_img + 1) / 2  # от [-1,1] к [0,1]

            img_tensor = gen_img.squeeze(0)
            img_array = img_tensor.permute(1, 2, 0).numpy()
            img_array = (img_array * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_array)

            img_pil = img_pil.resize((224, 224))
            self.tk_img = ImageTk.PhotoImage(img_pil)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        self.spinner.config(text="✅ Готово!")

# === Запуск ===
root = tk.Tk()
app = HotdogGeneratorApp(root)
root.mainloop()