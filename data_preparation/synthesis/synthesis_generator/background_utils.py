import os
import requests
from tqdm import tqdm


def download_backgrounds(backgrounds_dir: str, num_backgrounds: int = 1000, img_size: int = 1920, thematic: bool = False) -> None:
    """Download background images from Picsum or Unsplash API."""
    os.makedirs(backgrounds_dir, exist_ok=True)

    if thematic:
        # Use Unsplash API with thematic queries (requires API key)
        # For now, fallback to Picsum with larger size
        print("Thematic backgrounds require Unsplash API key. Using high-res Picsum instead.")
        img_size = 1920
        urls = [f"https://picsum.photos/{img_size}/{img_size}?random={i}" for i in range(num_backgrounds)]
    else:
        urls = [f"https://picsum.photos/{img_size}/{img_size}?random={i}" for i in range(num_backgrounds)]

    successful_downloads = 0
    for i, url in enumerate(tqdm(urls, desc="Downloading backgrounds")):
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                filename = f"bg_{i:04d}.jpg"
                filepath = os.path.join(backgrounds_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(response.content)
                successful_downloads += 1
            else:
                print(f"Failed to download {i}: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error downloading {i}: {e}")

    print(f"Downloaded {successful_downloads}/{num_backgrounds} backgrounds to {backgrounds_dir}")
    print(f"Image size: {img_size}x{img_size}, Thematic: {thematic}")