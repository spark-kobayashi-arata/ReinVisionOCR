import numpy as np
from dataclasses import dataclass
import itertools

from models import *


__all__ = [
    "ReinVisionOCR",
]


@dataclass
class Rows:
    indexes:list[int]
    
    point:int
    
    top:int
    bottom:int
    
    def __init__(
        self,
        spacing:int,
        index:int,
        point:int,
    ):
        self.indexes = [index]
        self.point = point
        self.top = point - spacing
        self.bottom = point + spacing
    
    def update(self, index:int, point:int) -> bool:
        if self.top <= point <= self.bottom:
            self.indexes.append(index)
            return True
        
        return False


def text_sorted(
    bboxes:list[tuple[int, int, int, int]],
    spacing:int,
) -> list[tuple[int, int, int, int]]:
    """テキストソート

    Args:
        bboxes (list[tuple[int, int, int, int]]): _description_
        spacing (int): _description_

    Returns:
        list[tuple[int, int, int, int]]: _description_
    """
    # x軸ソート
    bboxes = sorted(bboxes, key=lambda bbox:bbox[0])
    
    # 中心座標を算出
    points = [(idx, ymin + (ymax - ymin) // 2) for idx, (_, ymin, _, ymax) in enumerate(bboxes)]
    
    # 行リスト
    rows = [Rows(spacing, *points.pop(0))]
    
    for i in range(1, len(bboxes)):
        is_update = False
        
        # pointがrowの範囲内に含まれるか
        for row, (index, point) in itertools.product(rows, points):
            if row.update(index, point):
                is_update = True
                points.remove((index, point))
                break
        
        # 既存のrowsに含まれない場合は行を追加
        if not is_update:
            index, point = points.pop(0)
            rows.append(Rows(spacing, index, point))
    
    # 行リストのソート
    rows.sort(key=lambda row:row.point)
    
    # y軸ソート
    indexes = [-1 for i in range(len(bboxes))]
    for y, row in enumerate(rows):
        for index in row.indexes:
            indexes[index] = y
    
    bboxes = [bbox for _, bbox in sorted(zip(indexes, bboxes))]
    
    return bboxes


class ReinVisionOCR:
    def __init__(
        self,
        craft_pretrained_path:str,
        post_craft_pretrained_path:str,
        coatnet_pretrained_path:str,
    ):
        self.craft = CRAFT.load_from_pretrained(craft_pretrained_path)
        
        self.post_craft = PostCRAFT.load_from_pretrained(post_craft_pretrained_path)
        
        self.coatnet = CoAtNet.load_from_pretrained(coatnet_pretrained_path)
        
    def image_to_text(
        self,
        image:np.ndarray,
        binary_threshold:float=0.01,
        char_size:int=10,
        char_threshold:float=0.1,
        spacing:int=10,
    ) -> tuple[str, list[tuple[int, int, int, int], list[np.ndarray]]]:
        """入力画像からテキスト

        Args:
            image (np.ndarray): _description_
            binary_threshold (float, optional): _description_. Defaults to 0.01.
            char_size (int, optional): _description_. Defaults to 10.
            char_threshold (float, optional): _description_. Defaults to 0.1.
            spacing (int, optional): _description_. Defaults to 10.

        Returns:
            tuple[str, list[tuple[int, int, int, int], list[np.ndarray]]]: _description_
        """
        # 文字領域の抽出
        bboxes = self.craft(image, binary_threshold, char_size, char_threshold)
        
        # 領域が存在しない
        if len(bboxes) == 0:
            return "", bboxes, []
        
        # テキストを想定したソート
        bboxes = text_sorted(bboxes, spacing)
        
        # 文字ピクセルの抽出
        chars = self.post_craft(image, bboxes)
        
        # 文字分類
        text = self.coatnet(chars)
        
        return text, bboxes, chars
