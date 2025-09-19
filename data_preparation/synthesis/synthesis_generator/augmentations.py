import albumentations as A


def get_augmentation_pipeline():
    """Get the augmentation pipeline for synthetic data generation."""
    return A.Compose([
        A.RandomBrightnessContrast(p=0.7, brightness_limit=0.3, contrast_limit=0.3),
        A.GaussNoise(p=0.3),
        A.MotionBlur(p=0.2),
        A.HueSaturationValue(p=0.6, hue_shift_limit=15),
    ])