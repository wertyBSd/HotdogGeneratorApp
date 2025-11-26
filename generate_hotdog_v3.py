import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# === Параметры ===
image_size = 64
batch_size = 64
latent_dim = 100
epochs = 100
real_label_smooth = 0.9
output_dir = "generated_hotdogs"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Подготовка данных ===
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Загружаем весь датасет
full_dataset = datasets.ImageFolder(root="dataset", transform=transform)

# Индекс класса 'hotdog'
hotdog_class_idx = full_dataset.class_to_idx['hotdog']

# Фильтруем только изображения с меткой 'hotdog'
hotdog_only_dataset = [sample for sample in full_dataset if sample[1] == hotdog_class_idx]

# Создаём DataLoader
dataloader = torch.utils.data.DataLoader(hotdog_only_dataset, batch_size=batch_size, shuffle=True)

# === Генератор с Dropout ===
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256*8*8),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# === Дискриминатор ===
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# === Инициализация моделей и оптимизаторов ===
generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

# === Обучение ===
d_losses, g_losses = [], []

for epoch in range(epochs):
    for i, data in enumerate(dataloader):
        real_imgs = data[0].to(device)
        batch_size_curr = real_imgs.size(0)

        # Метки с label smoothing
        real_labels = torch.full((batch_size_curr, 1), real_label_smooth, device=device)
        fake_labels = torch.zeros(batch_size_curr, 1, device=device)

        # === Обучение дискриминатора ===
        optimizer_D.zero_grad()
        outputs_real = discriminator(real_imgs)
        d_loss_real = criterion(outputs_real, real_labels)

        z = torch.randn(batch_size_curr, latent_dim).to(device)
        fake_imgs = generator(z)
        outputs_fake = discriminator(fake_imgs.detach())
        d_loss_fake = criterion(outputs_fake, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        if torch.isnan(d_loss):
            print(f"[⚠️] NaN в D_loss на эпохе {epoch+1}")
            continue
        d_loss.backward()
        optimizer_D.step()

        # === Обучение генератора ===
        optimizer_G.zero_grad()
        outputs = discriminator(fake_imgs)
        g_loss = criterion(outputs, real_labels)
        if torch.isnan(g_loss):
            print(f"[⚠️] NaN в G_loss на эпохе {epoch+1}")
            continue
        g_loss.backward()
        optimizer_G.step()

    d_losses.append(d_loss.item())
    g_losses.append(g_loss.item())
    print(f"Epoch [{epoch+1}/{epochs}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

    # Сохранение изображений
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        gen_imgs = generator(z).detach().cpu()
        if torch.isnan(gen_imgs).any():
            print(f"[⚠️] NaN в изображении на эпохе {epoch+1}")
        else:
            gen_imgs = (gen_imgs + 1) / 2
            vutils.save_image(gen_imgs, os.path.join(output_dir, f"hotdog_gen_epoch_{epoch+1}.png"))

    # Сохранение графика потерь каждые 10 эпох
    if (epoch + 1) % 10 == 0:
        plt.figure(figsize=(10, 4))
        plt.plot(d_losses, label='D_loss')
        plt.plot(g_losses, label='G_loss')
        plt.title('DCGAN Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"loss_plot_epoch_{epoch+1}.png"))
        plt.close()

# === Сохранение модели генератора ===
model_path = "generator_hotdog_v0_1.pt"
torch.save(generator.state_dict(), model_path)
print(f"[✅] Модель генератора сохранена по пути: {os.path.abspath(model_path)}")