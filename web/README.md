# MNIST 交互演示网页

附加交互演示，用于答辩现场展示（实验数据与结果分析以汇报 PPT 为准，本页不重复）。

## 打开方式

**双击 `index.html` 即可**，无需任何服务器：

- 全部依赖（Tailwind、FontAwesome）已本地化到 `assets/vendor/`，断网环境可用；
- 模型权重通过 `<script>` 标签加载 `assets/model_weights.js`，file:// 协议下同样生效。

也可以用静态服务器打开：`python -m http.server 8000`，访问 http://localhost:8000。

## 功能

1. **模型积木与动态代码**：拖拽/增删网络层，实时生成带形状注释的 PyTorch 代码。
   卷积、池化（MaxPool/AvgPool）、BatchNorm、全连接均为独立积木；
   每层输入由上一层输出自动推导（首层固定接收 1×28×28 灰度图，不可改）；
   非法结构（卷积/池化放到展平之后、特征图过小仍池化、输出层不是 Linear→10）会被实时拦截或提示自动修正；
   超参数面板含数据增强开关（对应对比实验维度），生成的代码同步附带增强片段。
2. **手写数字实时识别**：画板手写 0~9，浏览器端用 JS 复现完整 CNN 前向传播（卷积→池化→全连接），
   输入自动缩放到 28×28 并按质心居中以对齐 MNIST 分布。
   **支持拖拽/点击上传 .pth 更换模型**：浏览器内直接解析 torch.save(model.state_dict()) 的 zip 格式文件
   （`assets/pth_loader.js`，无需后端），按权重形状自动推导结构并热替换；默认仍加载自训练 CNN，
   可随时点击"恢复默认模型"还原。

## 模型权重

`assets/model_weights.js` 由 `export_weights.py` 从 `../MNIST_Submission/model.pth`
（训练 100 epoch 的 CNN）导出。重新训练模型后，在本目录运行：

```
python export_weights.py
```

即可更新网页使用的权重（脚本会同时用 numpy 复现前向传播并与 PyTorch 输出比对，验证导出正确性）。
