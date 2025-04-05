import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from PIL import Image, ImageTk
import subprocess
import threading
import sys

def resource_path(relative_path):
    """Получаем абсолютный путь к ресурсу"""
    try:
        # PyInstaller создает временную папку и хранит путь в _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class GOSTAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор ГОСТ")
        self.root.geometry("800x600")
        
        # Устанавливаем стиль
        self.style = ttk.Style()
        self.style.configure('Custom.TButton', padding=20, font=('Helvetica', 12))
        
        # Устанавливаем пастельные цвета
        self.root.configure(bg='#E6F3FF')  # Светло-голубой фон
        self.style.configure('TFrame', background='#E6F3FF')
        self.style.configure('TButton', 
                           background='#FFE5E5',  # Светло-розовый
                           font=('Helvetica', 12))
        
        # Создаем директорию для временных файлов
        self.temp_dir = 'temp_pages'
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
            
        # Регистрируем функцию очистки при закрытии
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Создаем фрейм с фоном
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Создаем фрейм для кнопок
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(expand=True)
        
        # Создаем статус бар
        self.status_var = tk.StringVar()
        self.status_var.set("Готов к работе")
        self.status_bar = ttk.Label(self.root, 
                                  textvariable=self.status_var, 
                                  relief=tk.SUNKEN, 
                                  anchor=tk.W,
                                  background='#E6F3FF',
                                  font=('Helvetica', 10))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Переменные
        self.selected_file = None
        self.analysis_results = None
        self.is_analyzing = False
        
        # Создаем кнопки
        self.create_buttons()
        
    def create_buttons(self):
        # Создаем стиль для больших кнопок
        self.style.configure('Large.TButton', 
                           padding=30, 
                           font=('Helvetica', 14, 'bold'),
                           width=20)
        
        # Кнопка загрузки файла
        load_button = ttk.Button(self.button_frame, 
                               text="Загрузить файл", 
                               command=self.load_file,
                               style='Large.TButton')
        load_button.pack(pady=10)
        
        # Кнопка анализа файла
        analyze_button = ttk.Button(self.button_frame, 
                                  text="Запустить анализ", 
                                  command=self.run_analysis,
                                  style='Large.TButton')
        analyze_button.pack(pady=10)
        
        # Кнопка просмотра результатов
        view_button = ttk.Button(self.button_frame, 
                               text="Просмотреть результаты", 
                               command=self.show_results,
                               style='Large.TButton')
        view_button.pack(pady=10)
        
    def clear_temp_files(self):
        """Очищает все временные файлы"""
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                try:
                    os.remove(os.path.join(self.temp_dir, file))
                except Exception as e:
                    print(f"Ошибка при удалении {file}: {e}")
                    
    def on_closing(self):
        """Обработчик закрытия приложения"""
        try:
            self.clear_temp_files()
        finally:
            self.root.destroy()
            
    def load_file(self):
        """Загрузка DOCX файла"""
        try:
            filename = filedialog.askopenfilename(
                title="Выберите DOCX файл",
                filetypes=[("Word документы", "*.docx"), ("Все файлы", "*.*")]
            )
            
            if filename:
                if not filename.lower().endswith('.docx'):
                    messagebox.showerror("Ошибка", "Пожалуйста, выберите файл формата DOCX")
                    return
                    
                self.selected_file = filename
                self.status_var.set(f"Загружен файл: {os.path.basename(filename)}")
                messagebox.showinfo("Успешно", "Файл успешно загружен. Теперь вы можете запустить анализ.")
            else:
                self.status_var.set("Загрузка файла отменена")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при загрузке файла: {str(e)}")
            self.status_var.set("Ошибка при загрузке файла")
            
    def run_analysis(self):
        """Запуск анализа документа"""
        if self.is_analyzing:
            messagebox.showwarning("Предупреждение", "Анализ уже выполняется")
            return
            
        if not self.selected_file:
            messagebox.showerror("Ошибка", "Сначала выберите DOCX файл для анализа")
            return
            
        if not os.path.exists(self.selected_file):
            messagebox.showerror("Ошибка", "Выбранный файл не существует")
            return
            
        # Создаем окно с предупреждением
        warning_window = tk.Toplevel(self.root)
        warning_window.title("Анализ")
        warning_window.geometry("400x200")
        warning_window.transient(self.root)
        warning_window.grab_set()
        
        # Центрируем окно
        warning_window.geometry("+%d+%d" % (
            self.root.winfo_x() + self.root.winfo_width()//2 - 200,
            self.root.winfo_y() + self.root.winfo_height()//2 - 100
        ))
        
        # Добавляем сообщение
        message_label = ttk.Label(warning_window, 
                                text="Подождите, идет анализ документа...",
                                font=('Helvetica', 12))
        message_label.pack(pady=20)
        
        # Добавляем прогресс бар
        progress = ttk.Progressbar(warning_window, 
                                mode='indeterminate')
        progress.pack(pady=20, padx=20, fill=tk.X)
        progress.start()
        
        # Добавляем дополнительное сообщение
        info_label = ttk.Label(warning_window,
                             text="Это может занять некоторое время",
                             font=('Helvetica', 10))
        info_label.pack(pady=10)
        
        def run_analysis_thread():
            try:
                self.is_analyzing = True
                self.status_var.set("Выполняется анализ документа...")
                
                # Очищаем временные файлы перед анализом
                self.clear_temp_files()
                
                # Получаем пути к файлам
                if getattr(sys, 'frozen', False):
                    # Если запущено как exe
                    base_path = sys._MEIPASS
                else:
                    # Если запущено как скрипт
                    base_path = os.path.dirname(os.path.abspath(__file__))
                
                predict_script = os.path.join(base_path, 'predict.py')
                model_file = os.path.join(base_path, 'best_gost_classifier.pth')
                
                # Проверяем существование файлов
                if not os.path.exists(predict_script):
                    raise Exception(f"Файл predict.py не найден: {predict_script}")
                if not os.path.exists(model_file):
                    raise Exception(f"Файл модели не найден: {model_file}")
                
                # Создаем временную директорию для страниц
                temp_dir = os.path.join(base_path, 'temp_pages')
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                
                # Запускаем predict.py
                cmd = [sys.executable, predict_script, self.selected_file]
                print(f"Запуск команды: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        'MODEL_PATH': model_file,
                        'TEMP_DIR': temp_dir
                    }
                )
                
                # Выводим результат для отладки
                print("stdout:", result.stdout)
                print("stderr:", result.stderr)
                
                if result.returncode != 0:
                    raise Exception(f"Ошибка при анализе: {result.stderr}")
                
                # Обрабатываем результаты
                self.analysis_results = result.stdout
                
                # Закрываем окно предупреждения
                self.root.after(0, warning_window.destroy)
                
                # Показываем сообщение об успехе
                self.root.after(0, lambda: messagebox.showinfo(
                    "Готово",
                    "Анализ завершен. Нажмите 'Просмотреть результаты' для просмотра."
                ))
                
                self.status_var.set("Анализ завершен успешно")
                
            except Exception as e:
                print(f"Ошибка: {str(e)}")
                self.root.after(0, lambda: messagebox.showerror(
                    "Ошибка",
                    f"Ошибка при анализе: {str(e)}"
                ))
                self.status_var.set("Ошибка при анализе документа")
            finally:
                self.is_analyzing = False
                
        # Запускаем анализ в отдельном потоке
        analysis_thread = threading.Thread(target=run_analysis_thread)
        analysis_thread.daemon = True
        analysis_thread.start()
        
    def show_results(self):
        if not self.analysis_results:
            messagebox.showerror("Ошибка", "Сначала выполните анализ")
            return
            
        # Создаем окно результатов
        results_window = tk.Toplevel(self.root)
        results_window.title("Результаты анализа")
        results_window.geometry("600x400")
        
        # Создаем стиль для кнопки просмотра
        self.style.configure('ViewPage.TButton',
                           padding=20,
                           font=('Helvetica', 12),
                           width=30)  # Увеличиваем ширину кнопки
        
        # Создаем фрейм для списка страниц
        pages_frame = tk.Frame(results_window, bg='#E6F3FF')
        pages_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создаем Treeview для отображения результатов
        columns = ('page', 'result', 'confidence')
        tree = tk.ttk.Treeview(pages_frame, columns=columns, show='headings')
        
        # Настраиваем заголовки
        tree.heading('page', text='Страница')
        tree.heading('result', text='Результат')
        tree.heading('confidence', text='Уверенность')
        
        # Настраиваем ширину колонок
        tree.column('page', width=100)
        tree.column('result', width=200)
        tree.column('confidence', width=100)
        
        # Добавляем скроллбар
        scrollbar = tk.ttk.Scrollbar(pages_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Размещаем элементы
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Добавляем кнопку просмотра страницы
        view_button = ttk.Button(results_window, 
                               text="Просмотреть выбранную страницу", 
                               command=lambda: self.view_page(tree),
                               style='ViewPage.TButton')
        view_button.pack(pady=20)  # Увеличиваем отступ
        
        try:
            # Выводим сырые данные в консоль для отладки
            print("Raw analysis results:", self.analysis_results)
            
            # Парсим результаты и добавляем их в таблицу
            lines = self.analysis_results.split('\n')
            current_page = None
            current_result = None
            current_confidence = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if "страница" in line.lower():
                    if current_page and current_result:
                        tree.insert('', tk.END, values=(f"Страница {current_page}", current_result, 
                                  f"{current_confidence}%" if current_confidence else "N/A"))
                    # Извлекаем номер страницы из строки "Страница X:"
                    current_page = line.lower().replace("страница", "").strip().rstrip(':')
                    current_result = None
                    current_confidence = None
                elif "результат" in line.lower():
                    current_result = line.split(":", 1)[1].strip() if ":" in line else line
                elif "уверенность" in line.lower():
                    current_confidence = line.split(":", 1)[1].strip().replace("%", "") if ":" in line else line
            
            # Добавляем последнюю запись
            if current_page and current_result:
                tree.insert('', tk.END, values=(f"Страница {current_page}", current_result, 
                          f"{current_confidence}%" if current_confidence else "N/A"))
                          
            # Если нет данных в таблице, показываем сообщение
            if len(tree.get_children()) == 0:
                tree.insert('', tk.END, values=('Нет данных', 'Нет результатов анализа', 'N/A'))
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при обработке результатов: {str(e)}")
            print("Error processing results:", str(e))
            print("Results:", self.analysis_results)
            
    def view_page(self, tree):
        selected = tree.selection()
        if not selected:
            messagebox.showwarning("Предупреждение", "Выберите страницу")
            return
            
        # Получаем номер страницы из выбранной строки
        page_text = tree.item(selected[0])['values'][0]
        try:
            # Извлекаем номер страницы из текста "Страница X"
            page_num = page_text.split()[-1].strip().rstrip(':')
            
            # Формируем путь к временному файлу
            image_path = os.path.join(self.temp_dir, f"temp_page_{page_num}.png")
            
            if os.path.exists(image_path):
                try:
                    # Открываем окно с изображением
                    page_window = tk.Toplevel(self.root)
                    page_window.title(f"Страница {page_num}")
                    page_window.geometry("1000x800")
                    
                    # Загружаем изображение
                    image = Image.open(image_path)
                    
                    # Получаем размеры окна
                    window_width = 900
                    window_height = 700
                    
                    # Вычисляем пропорции
                    image_ratio = image.width / image.height
                    window_ratio = window_width / window_height
                    
                    if image_ratio > window_ratio:
                        # Если изображение шире
                        new_width = window_width
                        new_height = int(window_width / image_ratio)
                    else:
                        # Если изображение выше
                        new_height = window_height
                        new_width = int(window_height * image_ratio)
                    
                    # Изменяем размер с сохранением пропорций
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(image)
                    
                    # Создаем фрейм для изображения
                    frame = tk.Frame(page_window)
                    frame.pack(fill=tk.BOTH, expand=True)
                    
                    # Создаем канвас с полосами прокрутки
                    canvas = tk.Canvas(frame, width=window_width, height=window_height)
                    scrollbar_y = tk.ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
                    scrollbar_x = tk.ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
                    
                    # Настраиваем прокрутку
                    canvas.configure(xscrollcommand=scrollbar_x.set, yscrollcommand=scrollbar_y.set)
                    
                    # Размещаем элементы
                    scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
                    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
                    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                    
                    # Добавляем изображение на канвас
                    canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                    canvas.image = photo  # Сохраняем ссылку
                    
                    # Настраиваем область прокрутки
                    canvas.configure(scrollregion=canvas.bbox(tk.ALL))
                    
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Ошибка при открытии изображения: {str(e)}")
            else:
                messagebox.showerror("Ошибка", f"Изображение страницы {page_num} не найдено\nПуть: {image_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось определить номер страницы: {str(e)}")

def main():
    root = tk.Tk()
    app = GOSTAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 