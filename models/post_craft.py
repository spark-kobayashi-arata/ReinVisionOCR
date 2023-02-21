import numpy as np
import cv2
import torch
from torch import Tensor

from .craft_base import *


__all__ = [
    "PostCRAFT",
]


class PostCRAFT(CRAFTBase):
    INPUT_SIZE = 64
    
    def __init__(
        self,
        pretrained:bool=False,
        freeze:bool=False,
    ):
        super().__init__(pretrained, freeze)
    
    def create_inputs(
        self,
        image:np.ndarray,
        bboxes:list[tuple[int, int, int, int]],
    ) -> Tensor:
        """bboxesから入力画像を作成

        Args:
            image (np.ndarray): _description_
            bboxes (list[tuple[int, int, int, int]]): _description_

        Returns:
            tuple[np.ndarray, tuple[int]]: _description_
        """
        inputs = torch.stack([
            self.transform(
                cv2.resize(
                    image[ymin:ymax, xmin:xmax],
                    (self.INPUT_SIZE, self.INPUT_SIZE),
                    interpolation=cv2.INTER_LINEAR,
                ),
            )
            for xmin, ymin, xmax, ymax in bboxes
        ])
        if self.is_cuda:
            inputs = inputs.cuda()
        
        return inputs
    
    @torch.no_grad()
    def __call__(
        self,
        image:np.ndarray,
        bboxes:list[tuple[int, int, int, int]],
    ) -> list[np.ndarray]:
        """画像から文字ピクセルの抽出

        Args:
            image (np.ndarray): _description_
            bboxes (list[tuple[int, int, int, int]]): _description_

        Returns:
            list[np.ndarray]: _description_
        """
        # 入力データの作成
        inputs = self.create_inputs(image, bboxes)
        
        # 推論
        regions:Tensor = super().__call__(inputs)
        regions = regions.squeeze().cpu().detach().numpy()
        
        # uint8に変換
        regions:list[np.ndarray] = [np.clip(region * 255, 0, 255).astype(np.uint8) for region in regions]
        
        return regions
