#!/usr/bin/env python3
"""
Скрипт для исправления путей в COCO JSON файле аннотаций Label Studio.

Преобразует абсолютные пути к изображениям в относительные пути
для корректной работы с папкой images/.
"""

import json
import os
import argparse
from pathlib import Path


def fix_coco_paths(coco_file_path: str, images_dir: str = "images") -> None:
    """
    Исправляет пути к изображениям в COCO JSON файле.

    Args:
        coco_file_path: Путь к COCO JSON файлу
        images_dir: Имя папки с изображениями (по умолчанию 'images')
    """

    # Проверяем существование файла
    if not os.path.exists(coco_file_path):
        raise FileNotFoundError(f"Файл {coco_file_path} не найден")

    # Читаем COCO файл
    print(f"Читаем файл: {coco_file_path}")
    with open(coco_file_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # Получаем директорию с COCO файлом
    coco_dir = Path(coco_file_path).parent

    # Исправляем пути в разделе images
    if 'images' in coco_data:
        print(f"Найдено {len(coco_data['images'])} изображений для обработки")

        for i, image in enumerate(coco_data['images']):
            old_path = image['file_name']
            print(f"Обработка изображения {i+1}: {old_path}")

            # Извлекаем имя файла из пути
            filename = Path(old_path).name

            # Создаем новый относительный путь
            new_path = f"{images_dir}/{filename}"

            # Обновляем путь в данных
            image['file_name'] = new_path
            print(f"  Изменено: {old_path} -> {new_path}")

    # Сохраняем исправленный файл
    output_file = coco_dir / "refs_ls_coco_fixed.json"
    print(f"\nСохраняем исправленный файл: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)

    print("Готово! Файл успешно обработан.")


def main():
    parser = argparse.ArgumentParser(
        description="Исправление путей в COCO JSON файле аннотаций Label Studio"
    )
    parser.add_argument(
        "coco_file",
        help="Путь к COCO JSON файлу (например: data/tbank_official_logos/refs_ls_coco.json)"
    )
    parser.add_argument(
        "--images-dir",
        default="images",
        help="Имя папки с изображениями (по умолчанию: images)"
    )

    args = parser.parse_args()

    try:
        fix_coco_paths(args.coco_file, args.images_dir)
    except Exception as e:
        print(f"Ошибка: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())