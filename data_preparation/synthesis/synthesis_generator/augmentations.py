import albumentations as A


def get_augmentation_pipeline():
    """Get the legacy augmentation pipeline for synthetic data generation."""
    return A.Compose([
        A.RandomBrightnessContrast(p=0.7, brightness_limit=0.3, contrast_limit=0.3),
        A.GaussNoise(p=0.3),
        A.MotionBlur(p=0.2),
        A.HueSaturationValue(p=0.6, hue_shift_limit=15),
    ])


def get_background_aug_pipeline():
    """Get augmentation pipeline for backgrounds."""
    return A.Compose([
        A.RandomBrightnessContrast(p=0.5, brightness_limit=0.3, contrast_limit=0.3),
        A.GaussNoise(p=0.3, var_limit=(10, 50)),
        A.MotionBlur(p=0.2, blur_limit=3),
        A.GaussianBlur(p=0.2, blur_limit=3),
        A.Solarize(p=0.1, threshold=128),
        A.RandomShadow(p=0.3),
        A.JPEGCompression(p=0.4, quality_lower=70, quality_upper=95),
    ])


def get_neg_aug_pipeline():
    """Get augmentation pipeline for distractor objects (negatives)."""
    return A.Compose([
        A.HueSaturationValue(p=0.6, hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20),
        A.RandomRotate90(p=0.3),
        A.GaussNoise(p=0.4, var_limit=(5, 30)),
        A.Perspective(p=0.3, scale=(0.05, 0.1)),
        A.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        A.GaussianBlur(p=0.3, blur_limit=3),
    ])


def get_logo_aug_pipeline():
    """Get augmentation pipeline for logos."""
    return A.Compose([
        A.RandomBrightnessContrast(p=0.7, brightness_limit=0.2, contrast_limit=0.2),
        A.Rotate(p=0.5, limit=30),
        A.ShiftScaleRotate(p=0.4, shift_limit=0.1, scale_limit=0.2, rotate_limit=15),
        A.Perspective(p=0.3, scale=(0.05, 0.15)),
        A.HueSaturationValue(p=0.5, hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=15),
        A.GaussNoise(p=0.4, var_limit=(5, 20)),
        A.ElasticTransform(p=0.2, alpha=1, sigma=50),
    ])