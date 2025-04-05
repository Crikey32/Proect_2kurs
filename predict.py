import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import logging
from docx2pdf import convert
from pdf2image import convert_from_path
import tempfile
import shutil
import sys

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GOSTClassifier(nn.Module):
    def __init__(self):
        super(GOSTClassifier, self).__init__()
        
        # Первый блок свертки
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Второй блок свертки
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Третий блок свертки
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Полносвязные слои
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def convert_docx_to_pdf(docx_path, pdf_path):
    """Конвертирует DOCX в PDF"""
    try:
        convert(docx_path, pdf_path)
        return True
    except Exception as e:
        logging.error(f"Ошибка при конвертации {docx_path}: {str(e)}")
        return False

def convert_pdf_to_image(pdf_path, output_dir):
    """Конвертирует PDF в изображения"""
    try:
        # Конвертируем все страницы PDF
        images = convert_from_path(pdf_path)
        if images:
            # Сохраняем каждую страницу как отдельное изображение
            base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
            for i, image in enumerate(images):
                output_path = os.path.join(output_dir, f"{base_filename}_page_{i+1}.png")
                image.save(output_path, 'PNG')
                logging.info(f"Сохранена страница {i+1} из {len(images)}")
            return True
        return False
    except Exception as e:
        logging.error(f"Ошибка при конвертации PDF в изображение {pdf_path}: {str(e)}")
        return False

def analyze_document(docx_path):
    """Анализирует документ на соответствие ГОСТу"""
    # Используем директорию temp_pages для сохранения страниц
    output_dir = "temp_pages"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # Очищаем директорию перед использованием
        for file in os.listdir(output_dir):
            try:
                os.remove(os.path.join(output_dir, file))
            except Exception as e:
                logging.error(f"Ошибка при удалении файла {file}: {str(e)}")
    
    # Создаем временную директорию для промежуточных файлов
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = os.path.join(temp_dir, "temp.pdf")
        
        # Конвертируем DOCX в PDF
        if not convert_docx_to_pdf(docx_path, pdf_path):
            return "Ошибка при конвертации DOCX в PDF"
        
        # Конвертируем PDF в изображения напрямую в temp_pages
        if not convert_pdf_to_image(pdf_path, output_dir):
            return "Ошибка при конвертации PDF в изображения"
        
        # Загружаем модель
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GOSTClassifier().to(device)
        
        try:
            model.load_state_dict(torch.load('best_gost_classifier.pth', map_location=device))
            model.eval()
        except Exception as e:
            logging.error(f"Ошибка при загрузке модели: {str(e)}")
            return "Ошибка при загрузке модели"
        
        # Определяем трансформации для изображений
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Анализируем каждое изображение
        results = []
        for img_name in sorted(os.listdir(output_dir)):
            if img_name.endswith('.png'):
                img_path = os.path.join(output_dir, img_name)
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities).item()
                    confidence = probabilities[0][predicted_class].item() * 100
                    
                    # Получаем номер страницы из имени файла
                    page_num = img_name.split('_')[-1].replace('.png', '')
                    
                    results.append({
                        'page': page_num,
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'probabilities': probabilities[0].tolist()
                    })
        
        # Сортируем результаты по номеру страницы
        results.sort(key=lambda x: int(x['page']))
        
        # Формируем итоговый результат
        if not results:
            return "Не удалось проанализировать документ"
        
        # Определяем общий результат на основе всех страниц
        total_pages = len(results)
        compliant_pages = sum(1 for r in results if r['predicted_class'] == 1)
        avg_confidence = sum(r['confidence'] for r in results) / total_pages
        
        result = f"Результат анализа документа:\n"
        result += f"Всего страниц: {total_pages}\n"
        result += f"Страниц, соответствующих ГОСТу: {compliant_pages}\n"
        result += f"Страниц, не соответствующих ГОСТу: {total_pages - compliant_pages}\n"
        result += f"Средняя уверенность: {avg_confidence:.2f}%\n\n"
        
        # Добавляем детальную информацию по каждой странице
        result += "Детальная информация по страницам:\n"
        for r in results:
            result += f"\nСтраница {r['page']}:\n"
            result += f"Результат: {'Соответствует ГОСТу' if r['predicted_class'] == 1 else 'Не соответствует ГОСТу'}\n"
            result += f"Уверенность: {r['confidence']:.2f}%\n"
        
        return result

def main():
    # Проверяем, передан ли путь к файлу как аргумент
    if len(sys.argv) < 2:
        logging.error("Не указан путь к файлу")
        return
    
    # Получаем путь к файлу из аргументов командной строки
    docx_path = sys.argv[1]
    
    if not os.path.exists(docx_path):
        logging.error(f"Файл {docx_path} не найден")
        return
    
    # Анализируем документ
    result = analyze_document(docx_path)
    print(result)

if __name__ == "__main__":
    main() 