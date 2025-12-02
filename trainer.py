import os
from torch.utils.data import DataLoader
from dataset import Dataset, get_training_augmentation, get_validation_augmentation
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from model import CamVidModel
import torch

if __name__ == "__main__":
    architecture = "dpt_simple"
    encoder = "dinov3_encoder"  #

    # save path
    save_dir = f"./lightning_logs/{architecture}_{encoder}_focal"

    # dataset
    data_root = r"/home/zcy/chenrong/segment/dataset/"

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
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Some training hyperparameters
    EPOCHS = 50
    T_MAX = EPOCHS * len(train_loader)
    # Always include the background as a class
    OUT_CLASSES = len(train_dataset.CLASSES)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    model = CamVidModel(architecture, encoder, in_channels=3, out_classes=OUT_CLASSES)
    model = model.to(device)

    checkpoint_callback = ModelCheckpoint(monitor="valid_dataset_iou", dirpath=save_dir, save_top_k=1, mode="max", filename="best-checkpoint")
    trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1, accelerator="gpu", callbacks=[checkpoint_callback], default_root_dir=save_dir)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )