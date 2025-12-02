import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torchvision import transforms as t
import os
import cv2
import numpy as np
import albumentations as A
import pdb

# training set images augmentation
def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0
        ),
        A.PadIfNeeded(min_height=512, min_width=512, p=1),
        A.RandomCrop(height=512, width=512, p=0.5),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(512, 512),
    ]
    return A.Compose(test_transform)



class Dataset(BaseDataset):
    CLASSES = [
        "unlabelled",
        "TR",
        "NBU",
        "OBU",
        "WA",
    ]

    def __init__(self, images_dir, masks_dir=None, classes=None, augmentation=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        if masks_dir:
            self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        else:
            self.masks_fps = None

        # Always map background ('unlabelled') to 0
        # self.background_class = self.CLASSES.index("unlabelled")
        #
        # # If specific classes are provided, map them dynamically
        # if classes:
        #     self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        # else:
        #     self.class_values = list(range(len(self.CLASSES)))  # Default to all classes

        # Create a remapping dictionary: class value in dataset -> new index (0, 1, 2, ...)
        # Background will always be 0, other classes will be remapped starting from 1.
        # self.class_map = {self.background_class: 0}
        # self.class_map.update(
        #     {
        #         v: i + 1
        #         for i, v in enumerate(self.class_values)
        #         if v != self.background_class
        #     }
        # )

        self.augmentation = augmentation

    def __getitem__(self, i):
        # Read the image
        image_name = self.images_fps[i].split("/")[-1]
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        if self.masks_fps:
            mask = cv2.imread(self.masks_fps[i], 0)
        else:
            mask = None

        # Read the mask in grayscale mode
        if self.augmentation:
            if self.masks_fps:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]
            else:
                sample = self.augmentation(image=image)
                image = sample["image"]

        # pdb.set_trace()
        # image = image.transpose(2, 0, 1)
        image = t.ToTensor()(image)

        if self.masks_fps:
            return image, mask, image_name
        else:
            return image, image_name

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    data_root = r"/data2/zcy/project/segment/dataset/"

    x_train_dir = os.path.join(data_root, "train/images")
    y_train_dir = os.path.join(data_root, "train/mask")
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
    )

    x_valid_dir = os.path.join(data_root, "val/images")
    y_valid_dir = os.path.join(data_root, "val/mask")
    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
    )

    x_test_dir = os.path.join(data_root, "test/images")
    y_test_dir = os.path.join(data_root, "test/mask")
    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
    )

    # Change to > 0 if not on Windows machine
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    import pdb
    pdb.set_trace()