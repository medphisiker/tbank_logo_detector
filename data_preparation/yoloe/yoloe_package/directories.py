import os

def create_directories(output_dir, runs_dir):
    """Создание необходимых директорий.

    Parameters
    ----------
    output_dir : str
        Путь к выходной директории.
    runs_dir : str
        Путь к директории для результатов прогонов.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)