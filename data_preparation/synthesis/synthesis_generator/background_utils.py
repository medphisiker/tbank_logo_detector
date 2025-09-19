import os
import requests
from tqdm import tqdm


def download_backgrounds(backgrounds_dir: str, num_backgrounds: int = 500, img_size: int = 640) -> None:
    """Download random background images from Picsum."""
    os.makedirs(backgrounds_dir, exist_ok=True)

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