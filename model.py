import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class GOSTDataset(Dataset):
    def __init__(self, gost_dir, notgost_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Загружаем изображения, соответствующие ГОСТу (метка 1)
        for img_name in os.listdir(gost_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(gost_dir, img_name))
                self.labels.append(1)
        
        # Загружаем изображения, не соответствующие ГОСТу (метка 0)
        for img_name in os.listdir(notgost_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(notgost_dir, img_name))
                self.labels.append(0)
        
        logging.info(f"Загружено {len(self.image_paths)} изображений")
        logging.info(f"Из них {sum(self.labels)} соответствуют ГОСТу")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

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

def train_model(model, train_loader, num_epochs=50, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_acc = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        
        logging.info(f'Эпоха [{epoch+1}/{num_epochs}], Потери: {epoch_loss:.4f}, Точность: {epoch_acc:.2f}%')
        
        # Обновляем learning rate на основе точности
        scheduler.step(epoch_acc)
        
        # Сохраняем лучшую модель
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_gost_classifier.pth')
            logging.info(f'Сохранена лучшая модель с точностью {best_acc:.2f}%')
    
    logging.info(f'Обучение завершено. Лучшая точность: {best_acc:.2f}%')

def main():
    # Определяем трансформации для изображений
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Создаем датасет
    dataset = GOSTDataset('dataset/gost', 'dataset/notgost', transform=transform)
    
    # Создаем загрузчик данных
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Создаем модель
    model = GOSTClassifier()
    
    # Обучаем модель
    train_model(model, train_loader, num_epochs=70, learning_rate=0.001)

if __name__ == '__main__':
    main() 