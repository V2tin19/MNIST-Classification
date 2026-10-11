import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import my_dataset
from model import CNN

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(CURRENT_DIR, "实验图表输出")
os.makedirs(FIG_DIR, exist_ok=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataset = my_dataset(split='test')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)    

    model = CNN().to(device)
    model_path = os.path.join(CURRENT_DIR, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    all_targets = []
    all_pred = [] 
    error_samples = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _,preds = torch.max(outputs, 1)

            preds_cpu = preds.cpu().numpy()
            labels_cpu = labels.numpy()

            all_pred.extend(preds_cpu)  
            all_targets.extend(labels_cpu)

            for img, pred, target in zip(images.cpu(), preds_cpu, labels_cpu):
                if pred !=target and len(error_samples)<16:
                    error_samples.append((img,target,pred))

    all_targets = np.array(all_targets)
    all_pred = np.array(all_pred)

    accuracy = 100.0 * (all_pred == all_targets).sum() / len(all_targets)
    print(f"Test Accuracy: {accuracy:.2f}% (Errors: {(all_pred != all_targets).sum()}/{len(all_targets)})")

#画martrix
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(all_targets, all_pred):
        cm[t, p] += 1   

    plt.figure(figsize = (8,7))
    plt.imshow(cm)
    plt.title("Confusion Martrix")
    plt.colorbar()

    ticks = np.arange(10)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            val = cm[i, j]
            if val > 0:
                plt.text(j, i, val, ha="center", va="center", color="white" if val > thresh else "black")
    plt.tight_layout()
    cm_save_path = os.path.join(FIG_DIR, "confusion_matrix.png")
    plt.savefig(cm_save_path)
    print(f"Confusion matrix saved to {cm_save_path}")

    cm_errors = cm.copy()
    np.fill_diagonal(cm_errors, 0)

    flat_indices = np.argsort(cm_errors.flatten())[::-1]
    for rank in range(5):
        idx = flat_indices[rank]

        true_cls, pred_cls = divmod(idx, 10)
        count = cm_errors[true_cls, pred_cls]
        if count > 0:
            print(f"Rank {rank+1}: {count} samples were misclassified as {pred_cls} when they were {true_cls}")

    if(len(error_samples)>0):
        plt.figure(figsize=(10,10))
        for i, (img, target, pred) in enumerate(error_samples):
            plt.subplot(4,4,i+1)
            plt.imshow(img.squeeze().numpy(), cmap='gray')
            plt.title(f"True: {target}, Pred: {pred}", fontsize=10)
            plt.axis('off')

        plt.suptitle("Top 16 Error Samples (True vs Pred)", fontsize=13, fontweight='bold')
        plt.tight_layout()
        err_save_path = os.path.join(FIG_DIR, "task4_error_samples.png")
        plt.savefig(err_save_path)

        print(f"Error samples saved to {err_save_path}")

if __name__ == "__main__":
    main()