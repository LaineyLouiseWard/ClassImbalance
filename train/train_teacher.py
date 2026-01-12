"""
Train the OEM teacher model (EfficientNet-B4 U-Net) and save a .pth checkpoint
compatible with TeacherUNet.load_checkpoint().

Run:
  python train_teacher.py -c config/teacher/unet_oem.py
"""

import os
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from geoseg.utils.cfg import py2cfg
from geoseg.utils.metric import Evaluator


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config_path", type=Path, required=True)
    return p.parse_args()


class TeacherTrain(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.loss = config.loss
        self.classes = config.classes

        ignore_index = getattr(config, "ignore_index", None)
        self.train_evaluator = Evaluator(num_class=len(self.classes), ignore_index=ignore_index)
        self.val_evaluator = Evaluator(num_class=len(self.classes), ignore_index=ignore_index)

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        return self.net(x)

    def on_train_epoch_start(self):
        self.train_evaluator.reset()
        self.training_step_outputs = []

    def training_step(self, batch, batch_idx):
        img = batch["img"]
        mask = batch["gt_semantic_seg"]

        logits = self(img)
        loss = self.loss(logits, mask)
        pred = torch.argmax(logits, dim=1)

        self.training_step_outputs.append(
            {"loss": loss, "pred": pred.detach().cpu().numpy(), "target": mask.detach().cpu().numpy()}
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        for output in self.training_step_outputs:
            self.train_evaluator.add_batch(output["target"], output["pred"])

        IoU = self.train_evaluator.Intersection_over_Union()
        mIoU = np.nanmean(IoU)
        F1 = np.nanmean(self.train_evaluator.F1())
        OA = self.train_evaluator.OA()

        self.log("train_mIoU", mIoU, on_epoch=True, prog_bar=True)
        self.log("train_F1", F1, on_epoch=True, prog_bar=True)
        self.log("train_OA", OA, on_epoch=True, prog_bar=True)

        print(f"\ntrain: {{'mIoU': {mIoU}, 'F1': {F1}, 'OA': {OA}}}")
        self.training_step_outputs.clear()

    def on_validation_epoch_start(self):
        self.val_evaluator.reset()
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        img = batch["img"]
        mask = batch["gt_semantic_seg"]

        logits = self(img)
        loss = self.loss(logits, mask)
        pred = torch.argmax(logits, dim=1)

        self.validation_step_outputs.append(
            {"loss": loss, "pred": pred.detach().cpu().numpy(), "target": mask.detach().cpu().numpy()}
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        for output in self.validation_step_outputs:
            self.val_evaluator.add_batch(output["target"], output["pred"])

        IoU = self.val_evaluator.Intersection_over_Union()
        mIoU = np.nanmean(IoU)
        F1 = np.nanmean(self.val_evaluator.F1())
        OA = self.val_evaluator.OA()

        self.log("val_mIoU", mIoU, on_epoch=True, prog_bar=True)
        self.log("val_F1", F1, on_epoch=True, prog_bar=True)
        self.log("val_OA", OA, on_epoch=True, prog_bar=True)

        print(f"\nval: {{'mIoU': {mIoU}, 'F1': {F1}, 'OA': {OA}}}")
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return [self.config.optimizer], [self.config.lr_scheduler]


def main():
    args = get_args()
    config = py2cfg(args.config_path)

    seed_everything(42)
    pl.seed_everything(42, workers=True)

    seed = 42
    g = torch.Generator()
    g.manual_seed(seed)

    if hasattr(config, "train_loader") and config.train_loader is not None:
        config.train_loader.worker_init_fn = seed_worker
        config.train_loader.generator = g

    if hasattr(config, "val_loader") and config.val_loader is not None:
        config.val_loader.worker_init_fn = seed_worker
        config.val_loader.generator = g

    model = TeacherTrain(config)

    os.makedirs(config.weights_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.weights_path,
        filename=config.weights_name,
        monitor=config.monitor,
        mode=config.monitor_mode,
        save_top_k=config.save_top_k,
        save_last=config.save_last,
    )

    csv_logger = CSVLogger(save_dir="lightning_logs", name=config.log_name)
    tb_logger = TensorBoardLogger(save_dir="lightning_logs", name=config.log_name)

    trainer = pl.Trainer(
        max_epochs=config.max_epoch,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback],
        logger=[csv_logger, tb_logger],
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        log_every_n_steps=10,
        precision=16 if torch.cuda.is_available() else 32,
    )

    trainer.fit(model, config.train_loader, config.val_loader)

    # -------------------------
    # Export final teacher .pth
    # -------------------------
    # We save a plain state_dict .pth in the EXACT place your KD expects.
    out_pth = osp_join(config.weights_path, f"{config.weights_name}.pth")
    torch.save(model.net.model.state_dict(), out_pth)
    print(f"\nSaved teacher state_dict to: {out_pth}")
    print("This should be loadable by TeacherUNet.load_checkpoint(...).")


def osp_join(*parts):
    import os.path as osp
    return osp.join(*parts)


if __name__ == "__main__":
    main()
