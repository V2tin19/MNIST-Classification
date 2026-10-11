import os
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "实验图表输出")
os.makedirs(FIG_DIR, exist_ok=True)

# 数据集不随仓库分发：默认到工程上一级的 MNIST/Dataset 找，找不到请修改下方 BASE_DIR
DEFAULT_REL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "MNIST"))
if os.path.exists(os.path.join(DEFAULT_REL_DIR, "Dataset")):
    BASE_DIR = DEFAULT_REL_DIR
else:
    raise FileNotFoundError(
        "未找到 MNIST 数据集（需包含 Dataset/images 与 Dataset/labels）。"
        "请将数据集放置于 ../../MNIST，或修改 dataset.py 中的 BASE_DIR 指向数据集位置。"
    )

class my_dataset(Dataset):
    def __init__(self, split='train'):
        csv_path = os.path.join(BASE_DIR, "Dataset", "labels", split, "labels.csv")
        self.df = pd.read_csv(csv_path)
        self.transform = transforms.ToTensor()
        self.cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        
        img_path = os.path.join(BASE_DIR, "Dataset", self.df.iloc[idx, 0])
        label = int(self.df.iloc[idx, 1])
        img = Image.open(img_path)
        img = self.transform(img)
        self.cache[idx] = (img, label)
        return img, label

if __name__ == '__main__':
    # 16 random samples
    plt.figure(figsize=(6,6))
    random_indices = np.random.choice(len(my_dataset()), 16, replace=False)
    for i, idx in enumerate(random_indices):
        img, label = my_dataset()[idx]
        plt.subplot(4, 4, i+1)
        plt.imshow(img.squeeze().numpy(), cmap='gray')
        plt.title(f"Label: {label}", fontsize=10)
    plt.suptitle("MNIST 16 Random Samples", fontsize=13, fontweight="bold")
    plt.savefig(os.path.join(FIG_DIR, "task1_sample_16.png"))

    # Class distribution
    plt.figure(figsize=(8,4))
    counts = my_dataset().df['class'].value_counts().sort_index()
    plt.bar(counts.index, counts.values)
    plt.title("Class Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(range(10))
    for x, y in zip(counts.index, counts.values):
        plt.text(x, y + 50, str(y), ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task1_class_distribution.png"))
