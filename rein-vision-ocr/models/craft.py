import numpy as np
import torch
from torch import Tensor

from .craft_base import *
from .gaussian import *


__all__ = [
    "CRAFT",
]


class CRAFT(CRAFTBase):
    # 対応している入力サイズ
    INPUT_SIZE = (224, 864, 1024)
    
    def __init__(
        self,
        pretrained:bool=False,
        freeze:bool=False,
    ):
        super().__init__(pretrained, freeze)
    
    def create_input(self, image:np.ndarray) -> tuple[Tensor, tuple[int, int, int, int]]:
        """CRAFTへの入力画像を作成
        
        対応しているキャンバスサイズを作成後、画像を貼り付けている

        Args:
            image (np.ndarray): _description_

        Returns:
            tuple[Tensor, tuple[int, int, int, int]]: 入力画像とクリップ位置、元画像のサイズを返す
        """
        # キャンバスサイズの推定
        image_h, image_w = image.shape[:2]
        canvas_h = min([size for size in self.INPUT_SIZE if image_h <= size])
        canvas_w = min([size for size in self.INPUT_SIZE if image_w <= size])
        
        # 貼り付け位置
        paste_x = (canvas_w - image_w)//2
        paste_y = (canvas_h - image_h)//2
        
        # キャンバスに画像を貼り付け
        input = np.full((canvas_h, canvas_w, 3), 128, dtype=np.uint8)
        input[paste_y:paste_y+image_h, paste_x:paste_x+image_w] = image
        
        input:Tensor = self.transform(input)
        input = input.unsqueeze(0)
        if self.is_cuda:
            input = input.cuda()
        
        return input, (paste_x, paste_y, image_w, image_h)
    
    @torch.no_grad()
    def __call__(
        self,
        image:np.ndarray,
        binary_threshold:float=0.01,
        char_size:int=10,
        char_threshold:float=0.1,
    ) -> list[tuple[int, int, int, int]]:
        """画像から文字領域を検出

        Args:
            image (np.ndarray): 画像(RGB配置)
            binary_threshold (float, optional): _description_. Defaults to 0.01.
            char_size (int, optional): _description_. Defaults to 10.
            char_threshold (float, optional): _description_. Defaults to 0.1.

        Returns:
            list[tuple[int, int, int, int]]: _description_
        """
        # RGB画像のみ
        if len(image.shape) != 3 or image.shape[2] != 3:
            return []
        
        # 対応している最大サイズを超過している場合は不可
        if image.shape[0] > self.INPUT_SIZE[-1] or image.shape[1] > self.INPUT_SIZE[-1]:
            return []
        
        # 入力データの作成
        input, (x, y, w, h) = self.create_input(image)
        
        # 推論
        region:Tensor = super().__call__(input)
        region = region.squeeze().cpu().detach().numpy()
        
        # bbox抽出
        bboxes = region_to_bboxes(region, binary_threshold, char_size, char_threshold)
        bboxes = [(max(0, xmin-x), max(0, ymin-y), min(xmax-x, w), min(ymax-y, h)) for xmin, ymin, xmax, ymax in bboxes]
        
        return bboxes
