"""
Knowledge Distillation Training Script (KD).
Used for Stage 6 (final consolidation).

- Student: FT-UNetFormer
- Teacher: frozen external model
- Dataset + sampling fully defined in config
"""

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


# ---------------------------------------------------------------------
# Reproducibility (fair ablation)
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        type=Path,
        required=True,
        help="Path to KD config file",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------
class KDTrain(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Required objects from config
        self.net = config.net
        self.teacher = config.teacher
        self.loss = config.loss
        self.classes = config.classes

        # Freeze teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        ignore_index = getattr(config, "ignore_index", 255)

        self.train_evaluator = Evaluator(
            num_class=len(self.classes), ignore_index=ignore_index
        )
        self.val_evaluator = Evaluator(
            num_class=len(self.classes), ignore_index=ignore_index
        )

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        return self.net(x)

    # ---------------- TRAIN ----------------
    def on_train_epoch_start(self):
        self.train_evaluator.reset()
        self.training_step_outputs = []

    def training_step(self, batch, batch_idx):
        img = batch["img"]
        mask = batch["gt_semantic_seg"]

        student_logits = self(img)

        with torch.no_grad():
            teacher_logits = self.teacher(img)

        loss = self.loss(student_logits, mask, teacher_logits)

        pred = torch.argmax(student_logits, dim=1)

        self.training_step_outputs.append(
            {
                "loss": loss,
                "pred": pred.detach().cpu().numpy(),
                "target": mask.detach().cpu().numpy(),
            }
        )

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=img.size(0),
        )
        return loss

    def on_train_epoch_end(self):
        for o in self.training_step_outputs:
            self.train_evaluator.add_batch(o["target"], o["pred"])

        IoU = self.train_evaluator.Intersection_over_Union()
        F1 = self.train_evaluator.F1()
        OA = self.train_evaluator.OA()

        self.log("train_mIoU", np.nanmean(IoU), prog_bar=True)
        self.log("train_F1", np.nanmean(F1), prog_bar=True)
        self.log("train_OA", OA, prog_bar=True)

        print("\ntrain:",
              {"mIoU": np.nanmean(IoU), "F1": np.nanmean(F1), "OA": OA})
        print({self.classes[i]: IoU[i] for i in range(len(self.classes))})

        self.training_step_outputs.clear()

    # ---------------- VAL ----------------
    def on_validation_epoch_start(self):
        self.val_evaluator.reset()
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        img = batch["img"]
        mask = batch["gt_semantic_seg"]

        student_logits = self(img)
        with torch.no_grad():
            teacher_logits = self.teacher(img)

        loss = self.loss(student_logits, mask, teacher_logits)
        pred = torch.argmax(student_logits, dim=1)

        self.validation_step_outputs.append(
            {
                "loss": loss,
                "pred": pred.detach().cpu().numpy(),
                "target": mask.detach().cpu().numpy(),
            }
        )

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=img.size(0),
        )
        return loss

    def on_validation_epoch_end(self):
        for o in self.validation_step_outputs:
            self.val_evaluator.add_batch(o["target"], o["pred"])

        IoU = self.val_evaluator.Intersection_over_Union()
        F1 = self.val_evaluator.F1()
        OA = self.val_evaluator.OA()

        self.log("val_mIoU", np.nanmean(IoU), prog_bar=True)
        self.log("val_F1", np.nanmean(F1), prog_bar=True)
        self.log("val_OA", OA, prog_bar=True)

        print("\nval:",
              {"mIoU": np.nanmean(IoU), "F1": np.nanmean(F1), "OA": OA})
        print({self.classes[i]: IoU[i] for i in range(len(self.classes))})

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return [self.config.optimizer], [self.config.lr_scheduler]


# ---------------------------------------------------------------------
# Helper: load student weights only
# ---------------------------------------------------------------------
def _load_student_weights_from_pl_ckpt(model: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" not in ckpt:
        raise ValueError("Checkpoint missing state_dict")

    sd = {
        k.replace("net.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("net.")
    }

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Loaded student weights.")
    if missing:
        print("Missing (non-fatal):", missing[:10])
    if unexpected:
        print("Unexpected (non-fatal):", unexpected[:10])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = get_args()
    config = py2cfg(args.config_path)

    seed_everything(42)
    pl.seed_everything(42, workers=True)

    g = torch.Generator().manual_seed(42)
    if hasattr(config, "train_loader"):
        config.train_loader.worker_init_fn = seed_worker
        config.train_loader.generator = g
    if hasattr(config, "val_loader"):
        config.val_loader.worker_init_fn = seed_worker
        config.val_loader.generator = g

    model = KDTrain(config)

    if getattr(config, "pretrained_ckpt_path", None):
        print(f"Loading student weights from {config.pretrained_ckpt_path}")
        _load_student_weights_from_pl_ckpt(
            model.net, config.pretrained_ckpt_path
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.weights_path,
        filename=config.weights_name,
        monitor=config.monitor,
        mode=config.monitor_mode,
        save_top_k=config.save_top_k,
        save_last=config.save_last,
    )

    trainer = pl.Trainer(
        max_epochs=config.max_epoch,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback],
        logger=[
            CSVLogger("lightning_logs", name=config.log_name),
            TensorBoardLogger("lightning_logs", name=config.log_name),
        ],
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        log_every_n_steps=10,
        precision=16 if torch.cuda.is_available() else 32,
    )

    trainer.fit(model, config.train_loader, config.val_loader)


if __name__ == "__main__":
    main()
