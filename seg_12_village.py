import os
from torch.utils.data import DataLoader
from dataset import Dataset, get_training_augmentation, get_validation_augmentation
import pytorch_lightning as pl
from model import CamVidModel
import segmentation_models_pytorch as smp
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
import cv2
import json
import tqdm
import pdb

def compute_metrics(conf_matrix, ignore_background = False):
    num_classes = conf_matrix.shape[0]
    if ignore_background:
        include = [i for i in range(num_classes) if i != 0]
    else:
        include = [i for i in range(num_classes)]

    TP = np.diag(conf_matrix)
    FP = conf_matrix.sum(axis=0) - TP
    FN = conf_matrix.sum(axis=1) - TP
    TN = conf_matrix.sum() - (TP + FP + FN)

    IoU = TP / (TP + FP + FN + 1e-10)
    mIoU = np.nanmean(IoU[include])

    pixel_acc = TP.sum() / conf_matrix.sum()
    mean_acc = np.nanmean((TP / (conf_matrix.sum(axis=1) + 1e-10))[include])
    precision = np.nanmean((TP / (TP + FP + 1e-10))[include])
    recall = np.nanmean((TP / (TP + FN + 1e-10))[include])

    return {
        "mIoU": mIoU,
        "meanPA": mean_acc,
        "pixelAccuracy": pixel_acc,
        "meanPrecision": precision,
        "meanRecall": recall,
    }


def crf_post(image, prob):
    # pdb.set_trace()
    num_classes, w, h = prob.size()
    d = dcrf.DenseCRF2D(w, h, num_classes)
    unary = unary_from_softmax(prob.cpu().numpy())  # 转换为 CRF 要求的格式
    d.setUnaryEnergy(unary)

    # 添加像素间关系（颜色+位置）
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)
    Q = d.inference(5)
    refined = np.argmax(Q, axis=0).reshape((h, w))
    return refined


def postprocess_mask_with_edge_guidance(pr_mask, ori_image):
    """
    pr_mask: np.ndarray, shape=(H, W), 每个像素为类别索引
    ori_image: np.ndarray, shape=(H, W, 3), 原始图像
    """
    unique_classes = np.unique(pr_mask)
    final_mask = np.zeros_like(pr_mask)

    # 转为灰度做边缘检测
    gray = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    for cls in unique_classes:
        if cls == 0:  # 跳过背景
            continue

        class_mask = (pr_mask == cls).astype(np.uint8) * 255

        # Step 1: 去噪和平滑
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
        class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)

        # Step 2: 边缘引导裁剪
        aligned_mask = cv2.bitwise_and(class_mask, edges)

        # Step 3: 轮廓提取与拟合
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        smooth_mask = np.zeros_like(class_mask)

        for cnt in contours:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(smooth_mask, [approx], -1, 255, thickness=cv2.FILLED)

        # 融合原mask和边缘引导的裁剪结果
        combined = cv2.bitwise_or(smooth_mask, aligned_mask)

        final_mask[combined > 0] = cls

    return final_mask

if __name__ == "__main__":
    architecture = "dpt_simple"
    encoder = "dinov3_encoder_focal+dice"

    ckpt_path = fr"./lightning_logs/{architecture}_{encoder}/best-checkpoint.ckpt"
    crf = False
    post_process = False
    # ckpt_path = None

    # model
    model = CamVidModel(architecture, encoder, in_channels=3, out_classes=5)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True)['state_dict'])
    model = model.to("cuda")
    model.eval()

    # dataset
    data_root = r"/home/zcy/chenrong/segment/new_dataset_0629/ori_images/"
    # villages = ["zhangzhou", "wanan", "tuling", "tangdong", "qianhuangtu", "lingshui", "jinshan", "hongcuo", "dongzhang", "daigang", "chengping", "bigu"]

    save_dir = "/home/zcy/chenrong/segment/new_dataset_0629/segments/"


    for village in os.listdir(data_root):
        mask_dir = os.path.join(save_dir, f"test_{village}", "mask")
        os.makedirs(mask_dir, exist_ok=True)

        x_test_dir = os.path.join(data_root, f"{village}")
        y_test_dir = None
        test_dataset = Dataset(
            x_test_dir,
            y_test_dir,
            augmentation=get_validation_augmentation(),
        )

        # Change to > 0 if not on Windows machine
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

        # Some training hyperparameters
        OUT_CLASSES = len(test_dataset.CLASSES)

        hist = np.zeros((OUT_CLASSES, OUT_CLASSES))  # 用于累计混淆矩阵
        with tqdm.tqdm(total=len(test_loader), desc=f"Process {village}") as pbar:
            with torch.no_grad():
                for image, image_name in iter(test_loader):
                    # image: 1, 3, 512, 512
                    image = image.to("cuda")
                    # mask = mask.long()
                    logit = model(image)  # Get raw logits from the model
                    ori_image = cv2.imread(os.path.join(x_test_dir, image_name[0]))
                    if crf:
                        pr_mask = crf_post(ori_image, logit.softmax(dim=1)[0].cpu())
                    else:
                        pr_mask = logit.softmax(dim=1).argmax(dim=1).cpu()[0].numpy()  # Shape: [1, 512, 512]

                    if post_process:
                        pr_mask = postprocess_mask_with_edge_guidance(pr_mask, ori_image)
                    # pdb.set_trace()
                    # hist += confusion_matrix(mask[0].flatten().cpu().numpy(), pr_mask.flatten(), labels=np.arange(OUT_CLASSES))

                    pr_mask_image = Image.fromarray(pr_mask.astype(np.uint8))
                    pr_mask_image.save(os.path.join(mask_dir, os.path.splitext(image_name[0])[0] + ".png"))
                    pbar.update()