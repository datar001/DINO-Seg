import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch.optim import lr_scheduler


class CamVidModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            encoder_depth = 4,
            decoder_intermediate_channels=(256,512,1024,1024),
            decoder_fusion_channels=256,
            **kwargs,
        )

        # Preprocessing parameters for image normalization
        # params = smp.encoders.get_preprocessing_params(encoder_name)
        self.number_of_classes = out_classes
        self.register_buffer("std", torch.tensor([0.229,0.224,0.225]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor([0.485,0.456,0.406],).view(1, 3, 1, 1))

        # import pdb
        # pdb.set_trace()
        # freeze encoder
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        # Loss function for multi-class segmentation
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

        ignore_index = 255
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)
        self.focal = smp.losses.FocalLoss(mode=smp.losses.MULTICLASS_MODE, gamma=2.0, alpha=None,
                                          ignore_index=ignore_index)
        self.lovasz = smp.losses.LovaszLoss(mode=smp.losses.MULTICLASS_MODE, per_image=False, ignore_index=ignore_index)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # Normalize image
        # import pdb
        # pdb.set_trace()
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask, _ = batch

        # Ensure that image dimensions are correct
        assert image.ndim == 4  # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()

        # Mask shape
        assert mask.ndim == 3  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        dice_loss = self.loss_fn(logits_mask, mask)
        ce_loss = self.ce(logits_mask, mask)
        focal_loss = self.focal(logits_mask, mask)
        lovasz_loss = self.lovasz(logits_mask, mask)
        # import pdb
        # pdb.set_trace()
        loss = focal_loss  # dice_loss +

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)

        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode="multiclass", num_classes=self.number_of_classes
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # Per-image IoU and dataset IoU calculations
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        base_lr = 2e-4
        encoder_params = list(self.model.encoder.parameters())
        other_params = [p for name, p in self.model.named_parameters() if not name.startswith("encoder.")]

        optimizer = torch.optim.AdamW([
            # {'params': encoder_params, 'lr': base_lr * 0.1},
            {'params': other_params, 'lr': base_lr}
        ])
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        # import pdb
        # pdb.set_trace()
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

if __name__ == "__main__":
    model = CamVidModel("DeepLabV3plus", "resnet18", in_channels=3, out_classes=5)
    import pdb
    pdb.set_trace()