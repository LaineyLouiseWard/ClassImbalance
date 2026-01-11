"""Supervised (non-KD) Training Script for Stages 1–5."""

import os
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from geoseg.utils.cfg import py2cfg
from geoseg.utils.metric import Evaluator

def seed_worker(worker_id: int):
    # make dataloader workers deterministic
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=Path, required=True, help="Path to the config.")
    return parser.parse_args()


class SupervisionTrain(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.net = config.net
        self.loss = config.loss  # must accept (student_logits, targets)
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

        IoU_per_class = self.train_evaluator.Intersection_over_Union()
        mIoU = np.nanmean(IoU_per_class)
        F1_per_class = self.train_evaluator.F1()
        F1 = np.nanmean(F1_per_class)
        OA = self.train_evaluator.OA()

        self.log("train_mIoU", mIoU, on_epoch=True, prog_bar=True)
        self.log("train_F1", F1, on_epoch=True, prog_bar=True)
        self.log("train_OA", OA, on_epoch=True, prog_bar=True)

        print(f"\ntrain: {{'mIoU': {mIoU}, 'F1': {F1}, 'OA': {OA}}}")
        class_metrics = {self.classes[i]: IoU_per_class[i] for i in range(len(self.classes))}
        print(class_metrics)

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

        IoU_per_class = self.val_evaluator.Intersection_over_Union()
        mIoU = np.nanmean(IoU_per_class)
        F1_per_class = self.val_evaluator.F1()
        F1 = np.nanmean(F1_per_class)
        OA = self.val_evaluator.OA()

        self.log("val_mIoU", mIoU, on_epoch=True, prog_bar=True)
        self.log("val_F1", F1, on_epoch=True, prog_bar=True)
        self.log("val_OA", OA, on_epoch=True, prog_bar=True)

        print(f"\nval: {{'mIoU': {mIoU}, 'F1': {F1}, 'OA': {OA}}}")
        class_metrics = {self.classes[i]: IoU_per_class[i] for i in range(len(self.classes))}
        print(class_metrics)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return [self.config.optimizer], [self.config.lr_scheduler]


def _load_student_weights_from_pl_ckpt(model: nn.Module, ckpt_path: str):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise ValueError("Checkpoint has no 'state_dict'.")

    model_state = {k.replace("net.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("net.")}
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print("Loaded student weights.")
    if missing:
        print(f"Missing keys (non-fatal): {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"Unexpected keys (non-fatal): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")


def main():
    args = get_args()
    config = py2cfg(args.config_path)   # ← THIS LINE

    # ↓↓↓ ADD EVERYTHING BELOW RIGHT HERE ↓↓↓
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
    # ↑↑↑ END ADDITION ↑↑↑

    model = SupervisionTrain(config)

    # Optional: load pretrained weights (e.g., Stage 5 finetune after OEM pretrain)
    if getattr(config, "pretrained_ckpt_path", None):
        print(f"Loading pretrained student weights from: {config.pretrained_ckpt_path}")
        _load_student_weights_from_pl_ckpt(model.net, config.pretrained_ckpt_path)

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

    if getattr(config, "resume_ckpt_path", None):
        trainer.fit(model, config.train_loader, config.val_loader, ckpt_path=config.resume_ckpt_path)
    else:
        trainer.fit(model, config.train_loader, config.val_loader)


if __name__ == "__main__":
    main()
