import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import cv2
import pdb

# 假设有 N 张测试图像，y_pred 和 y_true 是 (N, H, W) 的 mask
# 这里有一个伪代码示范如何计算每个类别的像素占比

# y_pred: 预测的类别 mask (N, H, W)
# y_true: 手工标注的类别 mask (N, H, W)
# num_classes: 类别数，例如 5 类 (0-4)

pred_mask_dir = "./lightning_logs/deeplabv3+_resnet18/"
true_mask_dir = "../dataset/test"

num_classes = 5  # 类别数（假设你有 5 个类别，包括背景）

# 用于保存每个类别的预测占比和真实占比
y_pred_class = [[] for class_id in range(num_classes)]
y_true_class = [[] for class_id in range(num_classes)]

for img_name in os.listdir(os.path.join(true_mask_dir, "mask")):
    y_true = cv2.imread(os.path.join(true_mask_dir, "mask", img_name), 0)
    y_pred = cv2.imread(os.path.join(pred_mask_dir, "mask", img_name), 0)
    # pdb.set_trace()
    # 遍历每个类别，计算其在每张图片中的占比
    for class_id in range(num_classes):
        pred_class = (y_pred == class_id).astype(np.float32)
        true_class = (y_true == class_id).astype(np.float32)

        # 计算每个图像中该类的占比
        pred_class_ratio = np.mean(pred_class, axis=(0, 1))  # 每张图像的预测占比
        true_class_ratio = np.mean(true_class, axis=(0, 1))  # 每张图像的真实占比

        # 将结果保存
        y_pred_class[class_id].append(pred_class_ratio)
        y_true_class[class_id].append(true_class_ratio)

# 绘制每个类别的回归图
for class_id in range(num_classes):
    # 获取该类别的预测占比和真实占比
    y_pred_cls = np.array(y_pred_class[class_id])
    y_true_cls = np.array(y_true_class[class_id])

    # 拟合线性回归
    model = LinearRegression()
    model.fit(y_pred_cls.reshape(-1, 1), y_true_cls)
    y_fit = model.predict(y_pred_cls.reshape(-1, 1))

    # 绘图
    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred_cls, y_true_cls, label=f"Class {class_id}")
    plt.plot(y_pred_cls, y_fit, color='magenta')  # 拟合线
    plt.xlabel(f"Model Prediction (Class {class_id})")
    plt.ylabel(f"Manual Label (Class {class_id})")
    plt.title(f"R-Square: {r2_score(y_true_cls, y_fit):.5f}")
    plt.legend()
    # plt.show()
    save_path = os.path.join(pred_mask_dir, f"{class_id}_linear_regression.png")
    plt.savefig(save_path)
