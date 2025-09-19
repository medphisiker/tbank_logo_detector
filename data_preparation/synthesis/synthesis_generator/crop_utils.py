import os
import json
from PIL import Image
from pathlib import Path


def load_coco_annotations(coco_path: str) -> dict:
    """Load COCO annotations from JSON file."""
    with open(coco_path, 'r') as f:
        return json.load(f)


def crop_logos_from_annotations(coco: dict, images_dir: str, crops_dir: str) -> None:
    """Crop logos from images based on COCO annotations."""
    os.makedirs(crops_dir, exist_ok=True)

    img_dict = {img['id']: img for img in coco['images']}
    category_map = {cat['id']: cat['name'] for cat in coco['categories']}

    for ann in coco['annotations']:
        img_id = ann['image_id']
        img_info = img_dict[img_id]
        file_name = img_info['file_name']
        img_path = Path(images_dir) / Path(file_name).name
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            continue

        x, y, w, h = ann['bbox']
        category_id = ann['category_id']
        category_name = category_map[category_id]

        img = Image.open(img_path).convert("RGBA")
        crop = img.crop((x, y, x + w, y + h))

        crop_name = f"{category_name.replace('_', '')}_{img_id:02d}.png"
        crop_path = Path(crops_dir) / crop_name
        crop.save(crop_path)
        print(f"Cropped {crop_name} from {file_name}")

    print("Cropping completed.")


def crop_logos(coco_path: str, images_dir: str, crops_dir: str) -> None:
    """Main function to crop logos from COCO annotations."""
    coco = load_coco_annotations(coco_path)
    crop_logos_from_annotations(coco, images_dir, crops_dir)