import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

from dataset import my_dataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(CURRENT_DIR, "实验图表输出")
os.makedirs(FIG_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        # 卷积块 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32) 
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积块 2
        self.conv2 = nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(16)

        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.dropout1 = nn.Dropout(0.2) 
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(0.2) 
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # 展平分类
        x = self.flatten(x)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def main():
    print(f"训练设备: {device}")
    # 引入轻量数据增强
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=10),               # 随机旋转 ±10 度
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), # 随机微幅平移 5%
        transforms.ToTensor()
    ])

    full_dataset = my_dataset(split='train')
    full_dataset.transform = train_transform
    
    torch.manual_seed(42)
    train_data, val_data = random_split(full_dataset, [50000, 10000])
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    model = ImprovedCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 余弦退火学习率调度
    epochs = 30 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    best_val_acc = 0.0

    print("开始训练")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_train_acc = 100.0 * train_correct / train_total
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_train_acc)

        # 验证
        model.eval()
        val_running_loss = 0.0
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        epoch_val_loss = val_running_loss / len(val_loader)
        acc = 100.0 * val_correct / val_total
        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(acc)

        # 最优权重检查点
        if acc > best_val_acc:
            best_val_acc = acc
            best_model_path = os.path.join(CURRENT_DIR, "best_improved_model.pth")
            torch.save(model.state_dict(), best_model_path)
            star = "最优模型已保存"
        else:
            star = ""

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1:2d}/{epochs}] | lr: {current_lr:.6f} | Train Loss: {epoch_loss:.4f}, Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f}, Acc: {acc:.2f}%{star}")

    print(f"\n训练完毕！最高验证准确率达到: {best_val_acc:.2f}%")

    # 绘制对照曲线
    plt.figure(figsize=(11, 4.8))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Improved Train Loss', color='royalblue')
    plt.plot(val_loss_history, label='Improved Val Loss', color='darkorange', linestyle='--')
    plt.title('Improved Loss (Cosine Decay)', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Improved Train Acc', color='royalblue')
    plt.plot(val_acc_history, label='Improved Val Acc', color='darkgreen', linestyle='--')
    plt.title(f'Smoothed Accuracy (Peak Val: {best_val_acc:.2f}%)', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    curve_path = os.path.join(FIG_DIR, "improved_training_curves.png")
    plt.savefig(curve_path, dpi=200)
    print(f"进阶双线收敛曲线已保存至: {curve_path}")

if __name__ == '__main__':
    main()