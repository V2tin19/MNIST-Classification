import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import my_dataset
from model import MLP, CNN

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(CURRENT_DIR, "实验图表输出")
os.makedirs(FIG_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_experiment(model_type='CNN',opt_type='Adam',lr=0.001,epochs=10):
    full_dataset = my_dataset(split='train')
    torch.manual_seed(42)
    train_data, val_data = random_split(full_dataset, [50000, 10000])
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    if model_type == 'MLP':
        model = MLP().to(device)
    else:
        model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    if opt_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    val_acc_history = []

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 验证集评估
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        acc = 100.0 * correct / total
        val_acc_history.append(acc)
        print(f"  Epoch [{epoch+1:2d}/{epochs}] | 验证准确率: {acc:.2f}%")

    return val_acc_history

def main():
    epochs = 10
    # MLP vs CNN
    mlp_acc = train_experiment(model_type='MLP', opt_type='Adam', lr=0.001, epochs=epochs)
    cnn_adam_acc = train_experiment(model_type='CNN', opt_type='Adam', lr=0.001, epochs=epochs)
    #CNN:Adam vs SGD
    cnn_sgd_acc = train_experiment(model_type='CNN', opt_type='SGD', lr=0.01, epochs=epochs)
    #CNN,SGD:0.1 vs 0.001
    cnn_lr_01 = train_experiment(model_type='CNN', opt_type='SGD', lr=0.1, epochs=epochs)
    cnn_lr_001 = train_experiment(model_type='CNN', opt_type='SGD', lr=0.001, epochs=epochs)

    # 结果表格
    results = [
        ('MLP', 'Adam', 0.001, mlp_acc),
        ('CNN', 'Adam', 0.001, cnn_adam_acc),
        ('CNN', 'SGD', 0.01, cnn_sgd_acc),
        ('CNN', 'SGD', 0.1, cnn_lr_01),
        ('CNN', 'SGD', 0.001, cnn_lr_001),
    ]
    table_lines = ["===== 任务5 对比实验结果 (10 Epochs, 验证集 10000 张) ====="]
    for i, (name, opt, lr, acc) in enumerate(results, 1):
        table_lines.append(f"组{i}: {name} + {opt}(lr={lr}) | 最终验证准确率: {acc[-1]:.2f}% | 最佳: {max(acc):.2f}%")
    table_path = os.path.join(FIG_DIR, "task5_results_table.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines) + "\n")
    print("\n".join(table_lines))
    print(f"结果表格已保存至: {table_path}")

    #MLP vs CNN
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(mlp_acc, label='MLP (Adam)', color='blue')
    plt.plot(cnn_adam_acc, label='CNN (Adam)', color='orange')
    plt.title('MLP vs CNN with Adam', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    #Adam vs SGD
    plt.subplot(1, 3, 2)
    plt.plot(cnn_sgd_acc, label='CNN (SGD)', color='green')
    plt.plot(cnn_adam_acc, label='CNN (Adam)', color='orange')
    plt.title('CNN with SGD vs Adam', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    #lr
    plt.subplot(1, 3, 3)
    plt.plot(cnn_lr_01, label='lr = 0.1', color='red')
    plt.plot(cnn_sgd_acc, label='lr = 0.01', color='purple')
    plt.plot(cnn_lr_001, label='lr = 0.001', color='cyan')
    plt.title('CNN with Different Learning Rates', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "task5_comparison_charts.png")
    plt.savefig(save_path)
    print(f"Comparison charts saved to {save_path}")

if __name__ == '__main__':
    main()
