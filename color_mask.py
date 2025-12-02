import os
import tqdm
import cv2
import numpy as np


architecture = "dpt_simple"
encoder = "dinov3_encoder_focal+dice"
# villages = ["zhangzhou", "wanan", "tuling", "tangdong", "qianhuangtu", "lingshui", "jinshan", "hongcuo", "dongzhang", "daigang", "chengping", "bigu"]
image_root = r"/home/zcy/chenrong/segment/dataset/test/images"
mask_root = fr"/home/zcy/chenrong/segment/code/lightning_logs/{architecture}_{encoder}/test/"
# for type in os.listdir(image_root):
#     type_dir = os.path.join(image_root, type)
#     for village in os.listdir(type_dir):

test_dir = mask_root  # os.path.join(, type, f"test_{village}")
image_dir = image_root  # os.path.join(, type, f"{village}")
mask_dir = os.path.join(test_dir, "mask")
# import pdb
# pdb.set_trace()
overlayed_output_dir = os.path.join(test_dir, "overlayed")
os.makedirs(overlayed_output_dir, exist_ok=True)
color_mask_output_dir = os.path.join(test_dir, "color_mask")
os.makedirs(color_mask_output_dir, exist_ok=True)
with tqdm.tqdm(total=len(image_dir), desc=f"Process") as pbar:
    for image_name in os.listdir(image_dir):
        idx = os.path.splitext(image_name)[0]
        # 假设你已经有了 image 和 mask，image 是 RGB，mask 是灰度图（0-4）
        image_path = os.path.join(image_dir, str(idx) + ".png")
        mask_path = os.path.join(mask_dir, str(idx) + ".png")
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 定义颜色映射：1~4分别用不同颜色表示
        color_map = {
            1: (255, 0, 0),    # 红
            2: (0, 255, 0),    # 绿
            3: (0, 0, 255),    # 蓝
            4: (255, 255, 0),  # 青
        }

        # 创建一个彩色 mask 图像
        colored_mask = np.zeros_like(image)

        for label, color in color_map.items():
            colored_mask[mask == label] = color

        # 将彩色 mask 叠加到原图上（使用透明度 alpha）
        alpha = 0.5
        overlayed = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

        # 保存或显示
        cv2.imwrite(os.path.join(color_mask_output_dir, str(idx) + ".png"), colored_mask)
        cv2.imwrite(os.path.join(overlayed_output_dir, str(idx) + ".png"), overlayed)
        # 或用 matplotlib 显示
        # import matplotlib.pyplot as plt
        # plt.imshow(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()
        pbar.update()
