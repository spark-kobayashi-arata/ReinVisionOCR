from pathlib import Path
import pickle
import json

import torch
from torch import Tensor
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import warnings
from pytorch_lightning.utilities.warnings import PossibleUserWarning
# num_workers増やせ警告を非表示
warnings.simplefilter("ignore", PossibleUserWarning)

from models import *


class CoAtNetDataset(Dataset):
    MEAN_NORM = (0.5, )
    STD_NORM = (0.5, )
    
    def __init__(
        self,
        dataset_dirs:list[Path],
    ):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN_NORM, std=self.STD_NORM),
        ])
        
        self.image_paths:list[Path] = []
        for dataset_dir in dataset_dirs:
            self.image_paths.extend([path for path in dataset_dir.glob("*.pickle")])
    
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index:int):
        image_path = self.image_paths[index]
        
        with open(str(image_path), mode="rb") as f:
            image = pickle.load(f)
        
        label = int(image_path.stem.split("_")[1])
        
        image = self.transform(image)
        
        return image, label


class CoAtNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dirs:list[str],
        batch_size:int,
    ):
        super().__init__()
        
        self.save_hyperparameters()
    
    @property
    def num_classes(self) -> int:
        with open(str(Path(self.hparams.dataset_dirs[0]) / "codes.json"), mode="r", encoding="utf-8") as f:
            code_list = json.load(f)
            return len(code_list)
    
    def setup(self, stage:str) -> None:
        self.train_dataset = CoAtNetDataset([Path(dataset_dir) / "train" for dataset_dir in self.hparams.dataset_dirs])
        self.valid_dataset = CoAtNetDataset([Path(dataset_dir) / "valid" for dataset_dir in self.hparams.dataset_dirs])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, pin_memory=True, num_workers=2, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.hparams.batch_size, pin_memory=True, num_workers=2, shuffle=False)


class CoAtNetModule(pl.LightningModule):
    def __init__(
        self,
        
        # model
        image_size:tuple[int],
        in_channels:int,
        num_blocks:list[int],
        channels:list[int],
        num_classes:int,
        block_types:list[str]=["C", "C", "T", "T"],
        
        # loss
        label_smoothing:float=0.1,
        
        # scheduler
        first_cycle_steps:int=2,
        cycle_mult:float=1.0,
        max_lr:float=1.0e-03,
        min_lr:float=1.0e-05,
        warmup_steps:int=1,
        gamma:float=0.5,
        skip_first:bool=True,
    ):
        super().__init__()
        
        self.model = CoAtNet(image_size, in_channels, num_blocks, channels, num_classes, block_types)
        
        self.save_hyperparameters()

    def forward(self, x:Tensor):
        return self.model(x)

    def training_step(self, batch:list[Tensor, Tensor], batch_index:int):
        image, label = batch
        output = self(image)
        loss = F.cross_entropy(output, label, label_smoothing=self.hparams.label_smoothing)
        pred = (output.argmax(1) == label).type(torch.float)
        return {"loss":loss, "correct":pred}
        
    def training_epoch_end(self, outputs) -> None:
        train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        train_accuracy = torch.cat([x["correct"] for x in outputs]).mean()
        self.log("train_loss", train_loss)
        self.log("train_accuracy", train_accuracy)
    
    def validation_step(self, batch:list[Tensor, Tensor], batch_index:int):
        image, label = batch
        output = self(image)
        loss = F.cross_entropy(output, label, label_smoothing=self.hparams.label_smoothing)
        pred = (output.argmax(1) == label).type(torch.float)
        return {"val_loss":loss, "val_correct":pred}

    def validation_epoch_end(self, outputs):
        valid_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        valid_accuracy = torch.cat([x["val_correct"] for x in outputs]).mean()
        self.log("val_loss", valid_loss)
        self.log("val_accuracy", valid_accuracy)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.max_lr,
        )
        
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.hparams.first_cycle_steps,
            cycle_mult=self.hparams.cycle_mult,
            max_lr=self.hparams.max_lr,
            min_lr=self.hparams.min_lr,
            warmup_steps=self.hparams.warmup_steps,
            gamma=self.hparams.gamma,
        )
        if self.hparams.skip_first:
            scheduler.step()
        
        return [optimizer], [scheduler]


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    
    pl.seed_everything(522)
    
    dataset_dirs = [
        r"E:\ReinVisionOCR\coatnet\dataset\version_0",
    ]
    
    datamodule = CoAtNetDataModule(dataset_dirs, 64)
    
    model_version = 1
    
    if model_version == 1:
        num_blocks = [2, 2, 3, 5, 2]            # L
        channels = [64, 96, 192, 384, 768]      # D
    elif model_version == 2:
        num_blocks = [2, 2, 6, 14, 2]           # L
        channels = [128, 128, 256, 512, 1026]   # D
    elif model_version == 3:
        num_blocks = [2, 2, 6, 14, 2]           # L
        channels = [192, 192, 384, 768, 1536]   # D
    elif model_version == 4:
        num_blocks = [2, 2, 12, 28, 2]          # L
        channels = [192, 192, 384, 768, 1536]   # D
    else:
        assert False, "not support"
    
    model = CoAtNetModule((32, 32), 1, num_blocks, channels, datamodule.num_classes)
    
    logger = TensorBoardLogger(
        save_dir=r"E:\ReinVisionOCR\coatnet",
        name="log_logs",
        default_hp_metric=False,
    )
    
    callbacks = [
        LearningRateMonitor(
            log_momentum=False,
        ),
        ModelCheckpoint(
            monitor="val_accuracy",
            filename="checkpoint-{epoch}-{val_accuracy:.8f}-{val_loss:.8f}",
            save_top_k=3,
            mode="max",
            save_last=True,
        ),
    ]
    
    trainer = pl.Trainer(
        max_epochs=12,
        callbacks=callbacks,
        logger=logger,
        accelerator="gpu",
        devices=[0],
    )
    
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )
