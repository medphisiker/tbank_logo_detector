#!/usr/bin/env python3
"""
Скрипт для конвертации COCO аннотаций в формат Label Studio JSON.

Использует label-studio-converter для преобразования COCO файла в формат,
который можно импортировать в Label Studio.
"""

import subprocess
import sys
import json
from pathlib import Path

def check_and_install_converter():
    """Проверяет и устанавливает label-studio-converter если нужно."""
    try:
        # Проверяем, установлен ли label-studio-converter
        result = subprocess.run(['label-studio-converter', '--help'],
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("🔧 Устанавливаю label-studio-converter...")
            # Сначала пробуем uv sync
            install_result = subprocess.run(['uv', 'sync'], capture_output=True, text=True)
            
            if install_result.returncode != 0:
                # Если uv sync не сработал, пробуем uv add
                install_result = subprocess.run(['uv', 'add', 'label-studio-converter'],
                                              capture_output=True, text=True)
                
                if install_result.returncode != 0:
                    print(f"❌ Ошибка установки через uv: {install_result.stderr}")
                    return False
            
            print("✅ label-studio-converter успешно установлен!")
            return True
        return True
    except FileNotFoundError:
        print("🔧 label-studio-converter не найден. Устанавливаю через uv...")
        # Пробуем uv sync
        install_result = subprocess.run(['uv', 'sync'], capture_output=True, text=True)
        
        if install_result.returncode != 0:
            # Если uv sync не сработал, пробуем uv add
            install_result = subprocess.run(['uv', 'add', 'label-studio-converter'],
                                          capture_output=True, text=True)
            
            if install_result.returncode != 0:
                print(f"❌ Ошибка установки через uv: {install_result.stderr}")
                return False
        
        print("✅ label-studio-converter успешно установлен!")
        return True

def convert_coco_to_labelstudio(input_file: str, output_file: str) -> bool:
    """
    Конвертирует COCO аннотации в Label Studio JSON формат.

    Args:
        input_file: Путь к входному COCO файлу
        output_file: Путь к выходному Label Studio JSON файлу

    Returns:
        True если конвертация успешна, False в противном случае
    """
    try:
        # Проверяем существование входного файла
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"❌ Ошибка: Входной файл не найден: {input_file}")
            return False

        # Создаем директорию для выходного файла если нужно
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"🔄 Конвертация COCO файла: {input_file}")
        print(f"📁 Выходной файл: {output_file}")

        # Запускаем label-studio-converter
        result = subprocess.run([
            'label-studio-converter', 'import', 'coco',
            '-i', str(input_path),
            '-o', str(output_path)
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Конвертация завершена успешно!")
            print(f"📄 Создан файл: {output_file}")
            return True
        else:
            print(f"❌ Ошибка конвертации: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Ошибка выполнения: {str(e)}")
        return False


def fix_image_paths(json_file: str, dataset_name: str = "tbank_official_logos") -> bool:
    """
    Исправляет пути к изображениям в Label Studio JSON файле для соответствия Docker контейнеру.

    Args:
        json_file: Путь к JSON файлу с аннотациями
        dataset_name: Название датасета (по умолчанию "tbank_official_logos")

    Returns:
        True если исправление успешно, False в противном случае
    """
    try:
        json_path = Path(json_file)
        if not json_path.exists():
            print(f"❌ Ошибка: Файл не найден: {json_file}")
            return False

        # Читаем JSON файл
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Исправляем пути к изображениям
        corrected_count = 0
        for item in data:
            if 'data' in item and 'image' in item['data']:
                old_path = item['data']['image']
                # Извлекаем имя файла из старого пути
                if '\\images/' in old_path:
                    filename = old_path.split('\\images/')[-1]
                elif '/images/' in old_path:
                    filename = old_path.split('/images/')[-1]
                else:
                    # Если путь не содержит /images/, используем последнюю часть пути
                    filename = old_path.split('/')[-1] if '/' in old_path else old_path.split('\\')[-1]
                
                # Создаем новый путь в формате Label Studio local files
                new_path = f"/data/local-files/?d=label-studio/data/local/{dataset_name}/images/{filename}"
                item['data']['image'] = new_path
                corrected_count += 1

        # Сохраняем исправленный JSON файл
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ Исправлено путей к изображениям: {corrected_count}")
        return True

    except Exception as e:
        print(f"❌ Ошибка при исправлении путей: {str(e)}")
        return False

def main():
    """Основная функция скрипта."""
    print("🚀 Label Studio COCO Converter")
    print("=" * 40)

    # Проверяем и устанавливаем label-studio-converter
    if not check_and_install_converter():
        print("\n❌ Не удалось установить label-studio-converter. Установите его вручную:")
        print("   pip install label-studio-converter")
        sys.exit(1)

    # Пути к файлам (можно изменить при необходимости)
    input_file = "data/tbank_official_logos/refs_ls_coco.json"
    output_file = "data/tbank_official_logos/label_studio_annotations.json"

    # Конвертация
    success = convert_coco_to_labelstudio(input_file, output_file)

    if success:
        # Исправляем пути к изображениям
        print("\n🔧 Исправление путей к изображениям...")
        fix_success = fix_image_paths(output_file)
        
        if fix_success:
            print("✅ Пути к изображениям успешно исправлены!")
        else:
            print("⚠️  Не удалось исправить пути к изображениям. Файл может не работать корректно в Docker контейнере.")
        
        print("\n📋 Следующие шаги:")
        print("1. Откройте Label Studio в браузере")
        print("2. Перейдите в ваш проект")
        print("3. Нажмите Import > Upload Files")
        print(f"4. Загрузите файл: {output_file}")
        print("5. Аннотации будут импортированы вместе с изображениями")
    else:
        print("\n❌ Конвертация не удалась. Проверьте:")
        print("- Установлен ли label-studio-converter: uv sync или uv add label-studio-converter")
        print("- Корректность путей к файлам")
        print("- Формат входного COCO файла")
        sys.exit(1)

if __name__ == "__main__":
    main()
