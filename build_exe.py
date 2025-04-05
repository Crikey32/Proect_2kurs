import PyInstaller.__main__
import os
import sys
import torch
import shutil
from pathlib import Path

# Получаем текущую директорию
current_dir = os.getcwd()

# Проверяем наличие Poppler
poppler_path = os.path.join(current_dir, 'poppler')
if not os.path.exists(poppler_path):
    print("Poppler не найден. Пожалуйста, скачайте Poppler и распакуйте его в папку 'poppler' в текущей директории.")
    print("Скачать Poppler можно с: https://github.com/oschwartz10612/poppler-windows/releases/")
    sys.exit(1)

# Формируем команду для PyInstaller
command = [
    'gui_app.py',
    '--onefile',
    '--windowed',
    '--name=GOST_Analyzer',
    # Добавляем только необходимые файлы
    '--add-data', f'predict.py{os.pathsep}.',
    '--add-data', f'best_gost_classifier.pth{os.pathsep}.',
    # Добавляем Poppler
    '--add-data', f'poppler{os.pathsep}poppler',
    # PyTorch и зависимости
    '--hidden-import', 'torch',
    '--hidden-import', 'torch.nn',
    '--hidden-import', 'torch.nn.functional',
    '--hidden-import', 'torchvision.transforms',
    '--hidden-import', 'torchvision.models.resnet',
    # GUI и обработка изображений
    '--hidden-import', 'tkinter',
    '--hidden-import', 'tkinter.ttk',
    '--hidden-import', 'PIL',
    '--hidden-import', 'PIL.Image',
    '--hidden-import', 'PIL.ImageTk',
    # Конвертация документов
    '--hidden-import', 'pdf2image',
    '--hidden-import', 'docx2pdf',
    # Оптимизация
    '--noupx',  # Отключаем UPX для ускорения сборки
    '--clean',
    '--noconfirm'
]

# Добавляем иконку, если она существует
if os.path.exists('app_icon.ico'):
    command.extend(['--icon', 'app_icon.ico'])

# Запускаем PyInstaller
PyInstaller.__main__.run(command) 