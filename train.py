import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import my_dataset
from model import CNN

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(CURRENT_DIR, "实验图表输出")
os.makedirs(FIG_DIR, exist_ok=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = my_dataset('train')
    torch.manual_seed(42)
    # 50000 训练集，10000 验证集
    train_data, val_data = random_split(full_dataset, [50000, 10000])

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    total_epochs = 100
    best_val_acc = 0.0

    print("开始基线 CNN 训练 (100 Epochs)...")
    for epoch in range(total_epochs):
        # 1. 训练阶段
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
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100.0 * train_correct / train_total
        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc)

        # 2. 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = 100.0 * val_correct / val_total
        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == total_epochs - 1:
            print(f"Epoch [{epoch+1:3d}/{total_epochs}] | Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            model_save_path = os.path.join(CURRENT_DIR, "model.pth")
            torch.save(model.state_dict(), model_save_path)

    print(f"\n训练完成！最高验证准确率: {best_val_acc:.2f}%，模型已保存至 model.pth")

    # 3. 绘制双线对照收敛曲线 (识别是否过拟合)
    plt.figure(figsize=(12, 5))

    # 子图 1：损失对照 (Train Loss vs Val Loss)
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss', color='royalblue')
    plt.plot(val_loss_history, label='Val Loss', color='darkorange', linestyle='--')
    plt.title('Training & Validation Loss', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # 子图 2：准确率对照 (Train Acc vs Val Acc)
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Accuracy', color='royalblue')
    plt.plot(val_acc_history, label='Val Accuracy', color='darkorange', linestyle='--')
    plt.title('Training & Validation Accuracy', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    curve_save_path = os.path.join(FIG_DIR, "training_curves.png")
    plt.savefig(curve_save_path, dpi=200)
    print(f"训练与验证双线对照曲线已保存至: {curve_save_path}")

if __name__ == "__main__":
    main()
