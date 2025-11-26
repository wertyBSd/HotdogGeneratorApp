import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision.models as models

# === Параметры ===
image_size = 128
batch_size = 64
latent_dim = 100
epochs = 200
real_label_smooth = 0.8
output_dir = "generated_hotdogs"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Аугментация данных ===
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# === Загрузка и фильтрация датасета ===
full_dataset = datasets.ImageFolder(root="dataset", transform=transform)
hotdog_class_idx = full_dataset.class_to_idx['hotdog']
hotdog_only_dataset = [sample for sample in full_dataset if sample[1] == hotdog_class_idx]
dataloader = torch.utils.data.DataLoader(hotdog_only_dataset, batch_size=batch_size, shuffle=True)

# === ДОБАВЛЕНО: Классификаторы ===

img_size = 224
model_paths = {
    'ResNet50': 'hotdog_model_resnet.pt',
    'MobileNetV2': 'hotdog_model_mobilenet.pt',
    'EfficientNetB0': 'hotdog_model_efficientnet.pt'
}

transform_eval = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

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

models_list = []
for name, path in model_paths.items():
    if os.path.exists(path):
        models_list.append(load_model(name, path))
    else:
        print(f"[⚠️] Модель не найдена: {path}")

def predict_generated_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform_eval(img).unsqueeze(0).to(device)
    predictions = []
    for model in models_list:
        with torch.no_grad():
            output = model(img_tensor)
            predictions.append(output.item())
    avg = np.mean(predictions)
    return avg < 0.2  # True если HOTDOG 🌭

# === Генератор ===
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

# === Дискриминатор ===
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# === Инициализация ===
generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

fixed_z = torch.randn(1, latent_dim).to(device)

d_losses, g_losses = [], []
best_g_loss = float('inf')
no_improve_epochs = 0

# === Обучение ===
for epoch in range(epochs):
    for i, data in enumerate(dataloader):
        real_imgs = data[0].to(device)
        batch_size_curr = real_imgs.size(0)

        real_labels = torch.full((batch_size_curr, 1), real_label_smooth, device=device)
        fake_labels = torch.zeros(batch_size_curr, 1, device=device)

        # Обучение дискриминатора
        optimizer_D.zero_grad()
        outputs_real = discriminator(real_imgs)
        d_loss_real = criterion(outputs_real, real_labels)

        z = torch.randn(batch_size_curr, latent_dim).to(device)
        fake_imgs = generator(z)
        outputs_fake = discriminator(fake_imgs.detach())
        d_loss_fake = criterion(outputs_fake, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Обучение генератора
        optimizer_G.zero_grad()
        outputs = discriminator(fake_imgs)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

    d_losses.append(d_loss.item())
    g_losses.append(g_loss.item())
    print(f"Epoch [{epoch + 1}/{epochs}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

    # Ранняя остановка
    if g_loss.item() < best_g_loss:
        best_g_loss = g_loss.item()
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= 15:
            print(f"[⏹️] Остановка обучения: G_loss не улучшается 15 эпох подряд.")
            break

    # Генерация изображения
        # === Генерация изображения
        with torch.no_grad():
            generator.eval()
            gen_img = generator(fixed_z).detach().cpu()
            generator.train()

            gen_img = (gen_img + 1) / 2
            img_path = os.path.join(output_dir, f"hotdog_gen_epoch_{epoch + 1}.png")
            vutils.save_image(gen_img, img_path)

        # === Оценка изображения классификатором
        if predict_generated_image(img_path):
            success_count += 1
            print(f"[🌭] HOTDOG подтверждён ({success_count}/5)")
        else:
            success_count = 0
            print("[❌] Не похоже на хот-дог. Счётчик сброшен.")

        # === Условие остановки по классификатору
        if success_count >= 5:
            print("[✅] Генератор стабильно создаёт хот-доги. Остановка обучения.")
            break

    # Сохранение чекпоинта
    if (epoch + 1) % 10 == 0:
        torch.save(generator.state_dict(), os.path.join(output_dir, f"generator_epoch_{epoch + 1}.pt"))
        plt.figure(figsize=(10, 4))
        plt.plot(d_losses, label='D_loss')
        plt.plot(g_losses, label='G_loss')
        plt.title('GAN Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"loss_plot_epoch_{epoch + 1}.png"))
        plt.close()

# Финальное сохранение модели
model_path = "generator_hotdog_final.pt"
torch.save(generator.state_dict(), model_path)
print(f"[✅] Финальная модель сохранена: {os.path.abspath(model_path)}")