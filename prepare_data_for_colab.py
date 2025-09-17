#!/usr/bin/env python3
"""
Скрипт для подготовки данных проекта tbank_logo_detector для загрузки в Google Colab.

Читает конфигурацию из data_config.json, где указаны поддиректории data/ (data_sirius, tbank_official_logos, data_synt) и лимиты файлов (null для всей папки, число для подвыборки).
Создает отдельные ZIP архивы для указанных поддиректорий data/ (с возможной подвыборкой файлов для тестирования)
и перемещает их в папку tbank_logo_detector_data для удобной загрузки на Google Drive.
Это позволяет независимо обновлять и архивировать каждую поддиректорию с контролем объема данных.
"""

import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict


def create_archive_for_subdir(
    subdir: str,
    project_root: Path,
    colab_data_dir: Path,
    limit: Optional[int] = None
) -> Optional[str]:
    """
    Создает ZIP архив для указанной поддиректории, возможно с подвыборкой файлов.

    Args:
        subdir: Имя поддиректории (относительно project_root)
        project_root: Корневая директория проекта
        colab_data_dir: Директория для сохранения архивов
        limit: Максимальное количество файлов для включения (None для всех)

    Returns:
        Путь к созданному архиву или None, если не удалось создать
    """
    source_path = project_root / "data" / subdir

    if not (source_path.exists() and source_path.is_dir()):
        print(f"⚠ Поддиректория {subdir} не найдена или не является директорией, пропускаем")
        return None

    try:
        print(f"\nОбработка поддиректории: {subdir}")

        # Создаем имя архива с временной меткой
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_base_name = colab_data_dir / f"{subdir}_{timestamp}"

        if limit is None:
            # Архивируем всю директорию
            print(f"Создание архива: {zip_base_name}.zip")
            print(f"Исходная директория: {source_path}")
            
            shutil.make_archive(str(zip_base_name), 'zip', str(source_path))
            
            zip_path = zip_base_name.with_suffix('.zip')
        else:
            # Подвыборка файлов
            all_files = list(source_path.rglob('*'))
            files_to_include = [f for f in all_files if f.is_file()]
            
            num_files = len(files_to_include)
            if limit >= num_files:
                print(f"Лимит {limit} >= {num_files}, архивируем все файлы")
                limit = None  # Фолбэк на полный архив
                shutil.make_archive(str(zip_base_name), 'zip', str(source_path))
                zip_path = zip_base_name.with_suffix('.zip')
            else:
                random.seed(42)  # Для воспроизводимости
                selected_files = random.sample(files_to_include, limit)
                print(f"Выбрано {limit} файлов из {num_files} для архивирования")
                
                temp_dir = colab_data_dir / f"temp_{subdir}_{timestamp}"
                temp_dir.mkdir(exist_ok=True)
                
                for file_path in selected_files:
                    rel_path = file_path.relative_to(source_path)
                    dest_path = temp_dir / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, dest_path)
                    print(f"  Скопирован: {rel_path}")
                
                # Архивируем temp_dir
                shutil.make_archive(str(zip_base_name), 'zip', str(temp_dir))
                shutil.rmtree(temp_dir)  # Очищаем временную директорию
                
                zip_path = zip_base_name.with_suffix('.zip')

        archive_size = zip_path.stat().st_size / (1024 * 1024)  # Размер в МБ
        print(f"Размер архива: {archive_size:.1f} МБ")
        
        print(f"✓ Архив создан: {zip_path.name}")
        
        return str(zip_path)
        
    except Exception as e:
        print(f"✗ Ошибка при создании архива для {subdir}: {e}")
        return None


def prepare_data_for_colab():
    """
    Основная функция подготовки данных для Google Colab.

    Читает data_config.json и создает ZIP архивы на основе конфигурации.
    """
    project_root = Path.cwd()

    # Создаем папку для данных Colab
    colab_data_dir = project_root / "tbank_logo_detector_data"
    colab_data_dir.mkdir(exist_ok=True)
    print(f"Создана/использована папка: {colab_data_dir}")

    # Читаем конфигурацию
    config_path = project_root / "data_config.json"
    if not config_path.exists():
        print("❌ Файл data_config.json не найден, создайте его с конфигурацией поддиректорий")
        return []

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config: Dict[str, Optional[int]] = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Ошибка чтения data_config.json: {e}")
        return []

    created_archives = []

    for subdir, limit in config.items():
        archive = create_archive_for_subdir(subdir, project_root, colab_data_dir, limit=limit)
        if archive:
            created_archives.append(archive)

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
        print("❌ Не удалось создать ни одного архива (проверьте конфигурацию и наличие поддиректорий)")

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
            print("\n⚠ Подготовка данных завершена, но архивы не созданы (проверьте data_config.json)")
            return 0  # Не ошибка, если директории отсутствуют
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        return 1


if __name__ == "__main__":
    exit(main())