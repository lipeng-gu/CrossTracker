import os
import argparse
from datetime import datetime
from typing import Tuple, List

import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from tracker.config import cfg_from_yaml_file, cfg
from tracker.utils import set_random_seed, logger
from model.m3_model import Net
from dataset.kitti_dataset import KittiDataset


# ------------------------------------------------------------------------------
# 辅助函数：将 batch 数据移动到 device
# ------------------------------------------------------------------------------
def move_batch_to_device(batch: Tuple, device: torch.device) -> Tuple:
    img1, img2 = batch[0]
    pts1, pts2 = batch[1]
    geo1, geo2 = batch[2]
    labels = batch[3]

    return (
        img1.float().to(device),
        img2.float().to(device),
        pts1.transpose(2, 1).float().to(device),
        pts2.transpose(2, 1).float().to(device),
        geo1.transpose(2, 1).float().to(device),
        geo2.transpose(2, 1).float().to(device),
        labels.to(device),
    )


# ------------------------------------------------------------------------------
# Trainer 类，封装训练与验证逻辑
# ------------------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        set_random_seed(cfg.seed)

        # 设备和模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ngpu = torch.cuda.device_count()
        logger.info(f"Using device: {self.device}, GPUs: {self.ngpu}")

        self.model = Net(embed_dim=cfg.embed_dim).to(self.device)
        if self.ngpu > 1:
            self.model = nn.DataParallel(self.model)
        self.model.train()

        # 数据加载
        self.train_loader, self.val_loader = self._make_dataloaders()

        # 损失、优化器、调度器
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            self.model.parameters(),
            lr=cfg.train_lr,
            weight_decay=cfg.train_decay
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.train_epoch,
            eta_min=cfg.train_lr_min
        )

        # 保存路径
        os.makedirs(cfg.save_path, exist_ok=True)

    def _make_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_ds = KittiDataset(self.cfg, 'train')
        val_ds   = KittiDataset(self.cfg, 'val')
        return (
            DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True,
                       num_workers=self.cfg.num_workers, drop_last=True),
            DataLoader(val_ds,   batch_size=self.cfg.batch_size, shuffle=False,
                       num_workers=self.cfg.num_workers, drop_last=False),
        )

    def train(self) -> nn.Module:
        logger.info("========== Start Training ===========")
        for epoch in range(self.cfg.train_epoch):
            self._train_one_epoch(epoch)
            self.scheduler.step()
            self._save_checkpoint(epoch)
            self.validate()
        return self.model

    def _train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        lr = self.optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{self.cfg.train_epoch}, LR={lr:.2e}")
        progress = tqdm(self.train_loader, desc="Training", dynamic_ncols=True)

        for batch in progress:
            self.optimizer.zero_grad()
            img1, img2, pts1, pts2, geo1, geo2, label = move_batch_to_device(batch, self.device)

            outs = self.model(img1, geo1, pts1, img2, geo2, pts2)
            losses = [f * self.loss_fn(o, label) for o, f in zip(outs, (3, 1, 1))]
            loss = sum(l for l in losses)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress.set_postfix({
                f"l{i+1}": f"{loss.item():.4f}"
                for i, loss in enumerate(losses)
            })

        avg_loss = total_loss / len(self.train_loader)
        logger.info(f"  ▶ Training Loss: {avg_loss:.4f}")

    def _save_checkpoint(self, epoch: int):
        # filename = f"checkpoint_e{epoch}.pth"
        filename = f"checkpoint.pth"
        path = os.path.join(self.cfg.save_path, filename)
        state_dict = (
            self.model.module.state_dict()
            if self.ngpu > 1 else
            self.model.state_dict()
        )
        torch.save(state_dict, path)
        logger.info(f"  ▶ Saved checkpoint: {path}")

    def validate(self) -> None:
        self.model.eval()
        total_loss = 0.0
        all_labels: List[int] = []
        all_outs: List[List[List[float]]] = [[], [], []]

        progress = tqdm(self.val_loader, desc="Validation", dynamic_ncols=True)
        with torch.no_grad():
            for batch in progress:
                img1, img2, pts1, pts2, geo1, geo2, label = move_batch_to_device(batch, self.device)
                outs = self.model(img1, geo1, pts1, img2, geo2, pts2)
                for i, out in enumerate(outs):
                    all_outs[i].extend(out.cpu().tolist())
                all_labels.extend(label.cpu().tolist())

                losses = [f * self.loss_fn(o, label) for o, f in zip(outs, (3, 1, 1))]
                total_loss += sum(l for l in losses)

                progress.set_postfix({
                    f"l{i + 1}": f"{loss.item():.4f}"
                    for i, loss in enumerate(losses)
                })

        avg_loss = total_loss / len(self.val_loader)
        logger.info(f"  ▶ Validation Loss: {avg_loss:.4f}")

        # 计算并打印指标
        for idx, outs in enumerate(all_outs, start=1):
            # preds = [int(o[1] > o[0]) for o in outs]
            preds = torch.softmax(torch.tensor(outs), dim=-1)
            preds = (preds[:, 1] > self.cfg.sim_thr).int().tolist()
            p = precision_score(all_labels, preds, average='macro')
            r = recall_score(all_labels, preds, average='macro')
            f = f1_score(all_labels, preds, average='macro')
            cm = confusion_matrix(all_labels, preds)
            logger.info(f"    Modal {idx}: F1={f:.4f}, P={p:.4f}, R={r:.4f}, ConfMat={cm.tolist()}")

        self.model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrossTracker Training")
    parser.add_argument(
        "--cfg_file", type=str,
        default="config/kitti_train.yaml"
    )
    args = parser.parse_args()

    config = cfg_from_yaml_file(args.cfg_file, cfg)
    logger.info(f"Config loaded: {args.cfg_file}")
    logger.info(f"Start Time: {datetime.now()}")

    trainer = Trainer(config)
    trainer.train()

    logger.info(f"End Time:   {datetime.now()}")
