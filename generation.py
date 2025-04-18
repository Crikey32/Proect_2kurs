from docx import Document
import random

# Открываем документ
input_path = "Пояснительная записка.docx"
num_iterations = 10  # Количество итераций замены

# Словарь синонимов с расширенным списком ключей
synonyms = {
    "автобусов": ["транспортных средств", "маршрутов", "автотранспорта", "автобайков", "общественного транспорта"],
    "интервалов": ["промежутков", "отрезков", "интермиссий", "диапазонов", "пауз"],
    "данных": ["информации", "сведений", "фактов", "исходных материалов", "ресурсов"],
    "регулирования": ["контроля", "управления", "координации", "надзора", "мониторинга"],
    "расписания": ["графика", "плана", "тайминга", "таблицы времени", "регламента"],
    "движения": ["перемещения", "транспортировки", "потока", "течения", "маршрутизации"],
    "количество": ["число", "объем", "масштаб", "показатель", "размер"],
    "контейнер": ["емкость", "резервуар", "ящик", "бокс", "тару"],
    "учет": ["регистрация", "фиксирование", "мониторинг", "отслеживание", "контроль"],
    "операции": ["процессы", "действия", "манипуляции", "функции", "активности"],
    "подсистема": ["модуль", "составная часть", "раздел", "компонент", "элемент"],
    "анализ": ["исследование", "разбор", "диагностика", "оценка", "экспертиза"],
    "метод": ["способ", "подход", "техника", "прием", "алгоритм"],
    "разработка": ["создание", "конструирование", "проектирование", "формирование", "моделирование"],
    "эффективность": ["производительность", "результативность", "оптимальность", "профит", "функциональность"],
    "система": ["структура", "механизм", "платформа", "архитектура", "комплекс"],
    "оптимизация": ["улучшение", "совершенствование", "адаптация", "усовершенствование", "модернизация"],
    "участники": ["персонал", "команда", "группа", "сотрудники", "исполнители"],
    "требования": ["условия", "нормы", "критерии", "регламенты", "предписания"],
    "пользователь": ["оператор", "клиент", "заказчик", "администратор", "потребитель"],
    "интерфейс": ["панель", "GUI", "оболочка", "взаимодействие", "экранное представление"],
    "сервис": ["услуга", "поддержка", "обслуживание", "платформа", "программа"],
    "безопасность": ["защита", "конфиденциальность", "стабильность", "комплексная защита", "устойчивость"],
    "документация": ["бумаги", "отчеты", "инструкции", "архивы", "досье"],
    "автоматизация": ["механизация", "роботизация", "автономизация", "цифровизация", "технологизация"],
    "модуль": ["раздел", "составная часть", "компонент", "блок", "сегмент"],
    "проект": ["инициатива", "разработка", "план", "предложение", "программа"],
    "груз": ["товар", "продукция", "партия", "материалы", "поставки"],
    "терминал": ["пункт", "зона", "площадка", "станция", "хаб"],
    "склад": ["хранилище", "депо", "запасник", "логистический центр", "сток"],
    "доставка": ["транспортировка", "перевозка", "перемещение", "экспедирование", "логистика"],
    "контроль": ["мониторинг", "управление", "координация", "надзор", "регулирование"],
    "моделирование": ["проектирование", "создание схем", "визуализация", "построение модели", "структурирование"],
    "база данных": ["хранилище данных", "БД", "информационный массив", "структурированная информация", "бэкенд-хранилище"],
    "сервер": ["вычислительный центр", "главный узел", "узел сети", "хост", "серверная платформа"],
    "фильтр": ["критерий", "параметр", "условие", "отбор", "пресет"],
    "отчет": ["доклад", "резюме", "сводка", "информация", "итоговый документ"]
}

for i in range(1, num_iterations + 1):
    doc = Document(input_path)
    
    for para in doc.paragraphs:
        for run in para.runs:
            for word, variations in synonyms.items():
                if word in run.text:
                    run.text = run.text.replace(word, random.choice(variations))
    
    output_path = f"Пояснительная записка_плохие({i}).docx"
    doc.save(output_path)
    print(f"Итерация {i}: документ сохранен как {output_path}")
