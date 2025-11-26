import torch
import torchvision.utils as vutils
import os
import torch.nn as nn
from datetime import datetime

# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Размер латентного вектора
latent_dim = 100

# Класс генератора (соответствует обученной архитектуре)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Папка для сохранения
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

# Инициализация генератора
generator = Generator().to(device)
generator.load_state_dict(torch.load("generator_hotdog_v0_1.pt", map_location=device))
generator.eval()

# Генерация изображения
with torch.no_grad():
    z = torch.randn(1, latent_dim).to(device)  # [1, 100]
    gen_img = generator(z).detach().cpu()
    gen_img = (gen_img + 1) / 2  # от [-1,1] к [0,1]

    # Имя файла с меткой времени
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hotdog_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    # Сохранение изображения
    vutils.save_image(gen_img, filepath)
    print(f"[✅] Изображение сохранено: {filepath}")