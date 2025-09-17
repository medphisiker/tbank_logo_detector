import os
import requests
from tqdm import tqdm

# Конфигурация
backgrounds_dir = "data_preparation/synthesis/backgrounds"
num_backgrounds = 500
img_size = 640  # Размер изображений

os.makedirs(backgrounds_dir, exist_ok=True)

# Скачивание случайных фонов с Picsum (бесплатно, без API ключа; random photos для backgrounds)
for i in tqdm(range(num_backgrounds), desc="Downloading backgrounds"):
    url = f"https://picsum.photos/{img_size}/{img_size}?random={i}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(os.path.join(backgrounds_dir, f"bg_{i:04d}.jpg"), "wb") as f:
                f.write(response.content)
        else:
            print(f"Failed to download {i}")
    except Exception as e:
        print(f"Error downloading {i}: {e}")

print(f"Downloaded {num_backgrounds} backgrounds to {backgrounds_dir}")