#!/usr/bin/env python3
"""
Скрипт для подготовки данных проекта tbank_logo_detector для загрузки в Google Colab.

Создает архивы данных и перемещает их в специальную папку для удобной загрузки на Google Drive.
"""

import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime


def create_zip_archive(source_dir: str, zip_name: str, output_dir: str) -> str:
    """
    Создает ZIP архив из указанной директории.

    Args:
        source_dir: Путь к исходной директории
        zip_name: Имя ZIP файла (без расширения)
        output_dir: Директория для сохранения архива

    Returns:
        Путь к созданному архиву
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Директория {source_dir} не найдена")

    # Создаем имя архива с временной меткой
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"{zip_name}_{timestamp}.zip"
    zip_path = Path(output_dir) / zip_filename

    print(f"Создание архива: {zip_path}")
    print(f"Исходная директория: {source_path}")

    # Создаем ZIP архив
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in source_path.rglob('*'):
            if file_path.is_file():
                # Добавляем файл в архив с относительным путем
                arcname = file_path.relative_to(source_path.parent)
                zip_file.write(file_path, arcname)
                print(f"  Добавлен: {arcname}")

    archive_size = zip_path.stat().st_size / (1024 * 1024)  # Размер в МБ
    print(f"Размер архива: {archive_size:.1f} МБ")
    return str(zip_path)


def prepare_data_for_colab():
    """
    Основная функция подготовки данных для Google Colab.
    """
    project_root = Path.cwd()

    # Создаем папку для данных Colab
    colab_data_dir = project_root / "tbank_logo_detector_data"
    colab_data_dir.mkdir(exist_ok=True)
    print(f"Создана папка: {colab_data_dir}")

    # Список директорий для архивирования
    data_dirs = [
        ("data/tbank_official_logos", "tbank_official_logos"),
        ("data/data_sirius", "data_sirius")
    ]

    created_archives = []

    for source_dir, zip_name in data_dirs:
        source_path = project_root / source_dir

        if source_path.exists():
            try:
                print(f"\nОбработка директории: {source_dir}")
                zip_path = create_zip_archive(str(source_path), zip_name, str(colab_data_dir))
                created_archives.append(zip_path)
                print(f"✓ Архив создан: {Path(zip_path).name}")
            except Exception as e:
                print(f"✗ Ошибка при создании архива для {source_dir}: {e}")
        else:
            print(f"⚠ Директория {source_dir} не найдена, пропускаем")

    # Вывод итоговой информации
    print(f"\n{'='*50}")
    print("ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА")
    print(f"{'='*50}")

    if created_archives:
        print(f"\nСозданные архивы в папке {colab_data_dir}:")
        for archive in created_archives:
            archive_path = Path(archive)
            size_mb = archive_path.stat().st_size / (1024 * 1024)
            print(f"  - {archive_path.name}: {size_mb:.1f} МБ")
        print("\n📁 Загрузите папку tbank_logo_detector_data на Google Drive")
        print("📓 Затем подключите ее к Google Colab для работы с YOLOE и GROUNDING DINO")
    else:
        print("❌ Не удалось создать ни одного архива")

    return created_archives


def main():
    """Точка входа в скрипт."""
    print("🚀 НАЧАЛО ПОДГОТОВКИ ДАННЫХ ДЛЯ GOOGLE COLAB")
    print("=" * 50)

    try:
        archives = prepare_data_for_colab()
        if archives:
            print("\n✅ Подготовка данных завершена успешно!")
            return 0
        else:
            print("\n❌ Подготовка данных завершена с ошибками")
            return 1
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        return 1


if __name__ == "__main__":
    exit(main())