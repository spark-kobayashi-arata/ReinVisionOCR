import cv2
from pathlib import Path
import pickle

import torch
from torch import Tensor
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_tensor

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import warnings
from pytorch_lightning.utilities.warnings import PossibleUserWarning
# num_workers増やせ警告を非表示
warnings.simplefilter("ignore", PossibleUserWarning)

from models import *


class CRAFTDataset(Dataset):
    MEAN_NORM = (0.485, 0.456, 0.406)
    STD_NORM = (0.229, 0.224, 0.225)
    
    def __init__(
        self,
        dataset_dirs:list[Path],
        dratio:float,
    ):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN_NORM, std=self.STD_NORM),
        ])
        
        self.gaussian = GaussianGenerator(dratio=dratio)
        
        self.image_paths:list[Path] = []
        self.bboxes_paths:list[Path] = []
        for dataset_dir in dataset_dirs:
            self.image_paths.extend([path for path in dataset_dir.glob("*.png") if path.stem.startswith("image_")])
            self.bboxes_paths.extend([path for path in dataset_dir.glob("*.pickle") if path.stem.startswith("bboxes_")])
        
        assert(len(self.image_paths) == len(self.bboxes_paths))
    
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index:int):
        image_path = self.image_paths[index]
        bboxes_path = self.bboxes_paths[index]
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        
        with open(str(bboxes_path), mode="rb") as f:
            bboxes = pickle.load(f)
        
        region = self.gaussian(image.shape[1:], bboxes)
        region = cv2.resize(region, (image.shape[1]//2, image.shape[2]//2), interpolation=cv2.INTER_LINEAR)
        region = to_tensor(region)
        
        return image, region


class CRAFTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dirs:list[str],
        dratio:float,
        batch_size:int,
    ):
        super().__init__()
        
        self.save_hyperparameters()
    
    def setup(self, stage:str) -> None:
        self.train_dataset = CRAFTDataset([Path(dataset_dir) / "train" for dataset_dir in self.hparams.dataset_dirs], self.hparams.dratio)
        self.valid_dataset = CRAFTDataset([Path(dataset_dir) / "valid" for dataset_dir in self.hparams.dataset_dirs], self.hparams.dratio)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, pin_memory=True, num_workers=2, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.hparams.batch_size, pin_memory=True, num_workers=2, shuffle=False)


class CRAFTModule(pl.LightningModule):
    def __init__(
        self,
        
        # model
        pretrained:bool=True,
        freeze:bool=False,
        
        # optimizer
        lr:float=1.0e-03,
        
        # scheduler
        T_max:int=10,
        eta_min:float=1.0e-05,
    ):
        super().__init__()
        
        self.model = CRAFT(pretrained, freeze)
        
        self.save_hyperparameters()

    def forward(self, x:Tensor):
        return self.model(x)

    def training_step(self, batch:list[Tensor, Tensor], batch_idx:int):
        image, region = batch
        output = self(image)
        loss = F.mse_loss(output, region)
        return {"loss":loss}
    
    def training_epoch_end(self, outputs) -> None:
        train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", train_loss)
    
    def validation_step(self, batch:list[Tensor, Tensor], batch_idx:int):
        image, region = batch
        output = self(image)
        loss = F.mse_loss(output, region)
        return {"val_loss":loss}
    
    def validation_epoch_end(self, outputs):
        valid_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", valid_loss)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.T_max,
            eta_min=self.hparams.eta_min,
        )
        return [optimizer], [scheduler]


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    
    pl.seed_everything(522)
    
    dataset_dirs = [
        r"E:\ReinVisionOCR\craft\dataset\version_0",
    ]
    
    datamodule = CRAFTDataModule(dataset_dirs, 4.6, 3)
    
    model = CRAFTModule()
    
    logger = TensorBoardLogger(
        save_dir=r"E:\ReinVisionOCR\craft",
        name="log_logs",
        default_hp_metric=False,
    )
    
    callbacks = [
        LearningRateMonitor(
            log_momentum=False,
        ),
        ModelCheckpoint(
            monitor="val_loss",
            filename="checkpoint-{epoch}-{val_loss:.8f}",
            save_top_k=11,
            mode="min",
            save_last=True,
        ),
    ]
    
    trainer = pl.Trainer(
        max_epochs=11,
        callbacks=callbacks,
        logger=logger,
        accelerator="gpu",
        devices=[0],
    )
    
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )
