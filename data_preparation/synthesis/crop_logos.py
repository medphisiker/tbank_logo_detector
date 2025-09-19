from synthesis_generator.crop_utils import crop_logos
from pathlib import Path

# Paths
script_dir = Path(__file__).parent

# Determine paths based on environment (Docker vs local)
if Path("/app/data").exists():
    # Docker environment
    coco_path = "/app/data/tbank_official_logos/refs_ls_coco.json"
    images_dir = "/app/data/tbank_official_logos/images"
else:
    # Local environment
    coco_path = script_dir.parent.parent / "data" / "tbank_official_logos" / "refs_ls_coco.json"
    images_dir = script_dir.parent.parent / "data" / "tbank_official_logos" / "images"

crops_dir = script_dir / "crops"

crop_logos(str(coco_path), str(images_dir), str(crops_dir))