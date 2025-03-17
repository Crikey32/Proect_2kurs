import os
import pandas as pd
from docx import Document

def extract_formatting(doc_path):
    try:
        doc = Document(doc_path)
        paragraphs = doc.paragraphs
        
        if not paragraphs:
            return None
        
        first_para = paragraphs[0]
        font_name = first_para.runs[0].font.name if first_para.runs else None
        font_size = first_para.runs[0].font.size.pt if first_para.runs and first_para.runs[0].font.size else None
        
        line_spacing = first_para.paragraph_format.line_spacing if first_para.paragraph_format.line_spacing else None
        left_indent = first_para.paragraph_format.left_indent.pt if first_para.paragraph_format.left_indent else 0
        right_indent = first_para.paragraph_format.right_indent.pt if first_para.paragraph_format.right_indent else 0
        
        return {
            "file_name": os.path.basename(doc_path),
            "font_name": font_name,
            "font_size": font_size,
            "line_spacing": line_spacing,
            "left_indent": left_indent,
            "right_indent": right_indent
        }
    except Exception as e:
        print(f"Ошибка при обработке {doc_path}: {e}")
        return None

def process_folder(folder_path, label):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".docx"):
            file_path = os.path.join(folder_path, file)
            formatting_data = extract_formatting(file_path)
            if formatting_data:
                formatting_data["label"] = label
                data.append(formatting_data)
    return data

def create_dataset(correct_folder, incorrect_folder, output_csv="dataset.csv"):
    correct_data = process_folder(correct_folder, label=1)
    incorrect_data = process_folder(incorrect_folder, label=0)
    
    df = pd.DataFrame(correct_data + incorrect_data)
    df.to_csv(output_csv, index=False)
    print(f"Датасет сохранен в {output_csv}")

# Укажите пути к папкам с правильными и неправильными работами
correct_folder = "good"
incorrect_folder = "bad"

# Создаем датасет
create_dataset(correct_folder, incorrect_folder)