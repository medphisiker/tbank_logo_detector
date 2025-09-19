from synthesis_generator.background_utils import download_backgrounds
from pathlib import Path

# Конфигурация
script_dir = Path(__file__).parent
backgrounds_dir = script_dir / "backgrounds"
num_backgrounds = 10
img_size = 640  # Размер изображений

download_backgrounds(str(backgrounds_dir), num_backgrounds, img_size)