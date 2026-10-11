import torch, json, os, sys
import numpy as np
from PIL import Image
import pandas as pd

# 将上一级的 model.pth（最终提交的 CNN）导出为网页默认权重 assets/model_weights.js
# 并用 numpy 复现前向传播与 PyTorch 比对，验证导出正确性
WEB = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(WEB)  # mywork/

def find_model_path():
    candidates = [
        os.path.join(ROOT, "model.pth"),                     # mywork/model.pth（默认）
        os.path.join(ROOT, "..", "model.pth"),               # MNIST_Submission/model.pth
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("未找到 model.pth，请将本脚本放在 model.pth 的下级 web/ 目录中")

MODEL_PATH = find_model_path()
print("使用模型:", MODEL_PATH)

sd = torch.load(MODEL_PATH, map_location="cpu")

def r(o):
    return np.round(o.numpy(), 6).tolist()

w = {
    "arch": "CNNModel: conv1(1->32,k3,p1)+ReLU+MaxPool2x2 -> conv2(32->16,k5,p2)+ReLU+MaxPool2x2 -> flatten(784) -> fc1(120)+ReLU -> fc2(84)+ReLU -> fc3(10)",
    "source": os.path.relpath(MODEL_PATH, WEB),
    "conv1_w": r(sd["conv1.weight"]), "conv1_b": r(sd["conv1.bias"]),
    "conv2_w": r(sd["conv2.weight"]), "conv2_b": r(sd["conv2.bias"]),
    "fc1_w": r(sd["fc1.weight"]), "fc1_b": r(sd["fc1.bias"]),
    "fc2_w": r(sd["fc2.weight"]), "fc2_b": r(sd["fc2.bias"]),
    "fc3_w": r(sd["fc3.weight"]), "fc3_b": r(sd["fc3.bias"]),
}
out = os.path.join(WEB, "assets", "model_weights.js")
with open(out, "w", encoding="utf-8") as f:
    f.write("// 由最终提交模型导出的默认权重（PyTorch 训练 CNN）\n")
    f.write("// 通过 <script> 标签加载，file:// 双击打开亦可\n")
    f.write("window.MODEL_WEIGHTS = " + json.dumps(w, ensure_ascii=False) + ";\n")
print("model_weights.js size:", os.path.getsize(out), "bytes")

# ---- numpy 参考前向，与 torch 比对 ----
sys.path.insert(0, ROOT)
from model import CNN

model = CNN()
model.load_state_dict(sd)
model.eval()

# 测试集 labels.csv：优先相对工程目录，回退绝对路径（与 dataset.py 策略一致）
REL = os.path.normpath(os.path.join(ROOT, "..", "..", "MNIST"))
DATA_ROOT = REL
if not os.path.exists(os.path.join(REL, "Dataset")):
    print("警告：未找到 MNIST 数据集，跳过数值校验（仅导出权重）。")
    df = None
else:
    df = pd.read_csv(os.path.join(REL, "Dataset", "labels", "test", "labels.csv")).head(8)

def conv(inp, W, b, pad):
    oc, ic, K, _ = W.shape
    H, Wid = inp.shape[1], inp.shape[2]
    p = np.zeros((ic, H + 2 * pad, Wid + 2 * pad), dtype=np.float32)
    p[:, pad:pad + H, pad:pad + Wid] = inp
    out = np.zeros((oc, H, Wid), dtype=np.float32)
    for o in range(oc):
        for i in range(H):
            for j in range(Wid):
                out[o, i, j] = np.sum(p[:, i:i + K, j:j + K] * W[o]) + b[o]
    return out

def pool(inp):
    C, H, W_ = inp.shape
    return inp[:, : H // 2 * 2, : W_ // 2 * 2].reshape(C, H // 2, 2, W_ // 2, 2).max(axis=(2, 4))

def np_forward(x):  # x: (28,28)
    xx = x[None, :, :]
    c1 = np.maximum(conv(xx, np.array(w["conv1_w"], np.float32), np.array(w["conv1_b"], np.float32), 1), 0)
    p1 = pool(c1)
    c2 = np.maximum(conv(p1, np.array(w["conv2_w"], np.float32), np.array(w["conv2_b"], np.float32), 2), 0)
    p2 = pool(c2)
    v = p2.flatten()
    h1 = np.maximum(np.array(w["fc1_w"], np.float32) @ v + np.array(w["fc1_b"], np.float32), 0)
    h2 = np.maximum(np.array(w["fc2_w"], np.float32) @ h1 + np.array(w["fc2_b"], np.float32), 0)
    return np.array(w["fc3_w"], np.float32) @ h2 + np.array(w["fc3_b"], np.float32)

if df is not None:
    ok = 0
    for _, row in df.iterrows():
        path = os.path.join(DATA_ROOT, "Dataset", row["filename"])
        img = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        with torch.no_grad():
            t_logits = model(torch.from_numpy(img[None, None, :, :]))
        n_logits = np_forward(img)
        diff = np.abs(t_logits.numpy().flatten() - n_logits).max()
        t_pred, n_pred = int(np.argmax(t_logits.numpy())), int(np.argmax(n_logits))
        same = t_pred == n_pred == int(row["class"])
        ok += same
        print(f"label={row['class']} torch_pred={t_pred} numpy_pred={n_pred} max_diff={diff:.2e} all_match={same}")
    print(f"({ok}/8 三方一致)")
