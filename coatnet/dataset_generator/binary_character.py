from dataclasses import dataclass
import os
from pathlib import Path
from typing import Union
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import random
import pickle
import cv2
from fontTools.ttLib import TTFont
import itertools

from common import *


__all__ = [
    "BinaryCharacterConfig",
    "create_binary_character_image",
]


@dataclass
class BinaryCharacterConfig:
    """2値化文字生成の設定
    """
    max_workers:int
    
    output_dir:Path
    output_version:int
    
    font_list:tuple[ImageFont.FreeTypeFont, ...]
    cmap:tuple[int, ...]
    
    text_pos_range:tuple[int, int]
    text_pos_list:tuple[tuple[int, int]]
    
    resize:tuple[int, int]
    
    n_valid:int
    
    code_list:tuple[str, ...]
    
    def __init__(
        self,
        max_workers:int,
        output_dir:str,
        font_path:str,
        font_size:list[int],
        text_pos_pattern_list:list[int],
        resize:list[int],
        n_valid:int,
        ignore_code_list:list[list[str]],
        code_params:list[list[Union[int, str]]],
    ):
        self.max_workers = os.cpu_count() if max_workers == -1 else max_workers
        
        self.output_dir = Path(output_dir)
        self.output_version = [int(dir.stem.split("_")[1]) for dir in self.output_dir.glob("**") if dir.name.startswith("version_")]
        self.output_version = max(self.output_version) + 1 if len(self.output_version) > 0 else 0
        self.output_dir = self.output_dir / f"version_{self.output_version}"
        
        self.font_list = tuple([ImageFont.truetype(font_path, size) for size in font_size])
        self.cmap = tuple(TTFont(font_path).getBestCmap())
        
        self.text_pos_range = min(text_pos_pattern_list), max(text_pos_pattern_list) + 1
        self.text_pos_list = [(p, p) for p in text_pos_pattern_list]
        self.text_pos_list.extend(itertools.combinations(text_pos_pattern_list, 2))
        
        assert is_vector(resize, 2), "'resize' requires list[int] type."
        self.resize = tuple(resize)
        
        self.n_valid = n_valid
        
        self.code_list = self.create_code_list(code_params, ignore_code_list)
    
    @property
    def random_text_pos(self) -> tuple[int, int]:
        """ランダムなテキスト描画位置を取得

        Returns:
            tuple[int, int]: _description_
        """
        return random.randrange(*self.text_pos_range), random.randrange(*self.text_pos_range)
    
    def create_code_list(self, code_params:list[list[Union[int, str]]], ignore_code_list:list[list[str]]) -> tuple[str, ...]:
        """生成文字リストの作成

        Args:
            code_params (list[list[Union[int, str]]]): _description_

        Returns:
            tuple[str, ...]: _description_
        """
        # 変換テーブルの作成
        ignore = str.maketrans({char:None for ignore_code in ignore_code_list for idx, char in enumerate(ignore_code) if idx != 0})
        empty = {
            " ":None,
            "　":None,
            "\n":None,
            "\u3000":None,
        }
        table = str.maketrans(HALF2FULL | empty)
        
        text = ""
        
        for params in code_params:
            if is_vector(params, 2, int):
                params[1] += 1 # end+1
                for code in range(*params):
                    text += chr(code)
            elif is_vector(params, 1, str):
                if os.path.isfile(params[0]):
                    with open(params[0], mode="r", encoding="utf-8") as f:
                        text += f.read()
                else:
                    text += params[0]
            else:
                assert False, "'code_params' contains unsupported elements."
        
        # 半角to全角
        # スペースと改行の削除
        text = text.translate(table)
        
        # 類似文字の削除
        text = text.translate(ignore)
        
        # 未対応文字と重複削除
        code_list = set([code for char in text if (code:=ord(char)) in self.cmap])
        code_list = sorted(code_list)
        code_list = tuple([chr(code) for code in code_list])
        return code_list

    def create_train_iter(self):
        return iter([
            (
                text_pos,
                code,
                label,
                font,
                self.resize,
            )
            for label, code in enumerate(self.code_list)
            for text_pos in self.text_pos_list
            for font in self.font_list
        ])
    
    def create_valid_iter(self):
        return iter([
            (
                self.random_text_pos,
                code,
                label,
                font,
                self.resize,
            )
            for label, code in enumerate(self.code_list)
            for n in range(self.n_valid)
            for font in self.font_list
        ])

    def create_iter(self, stage:StageType):
        if stage is StageType.TRAIN:
            return self.create_train_iter()
        elif stage is StageType.VALID:
            return self.create_valid_iter()
        else:
            assert False, "not support."


def create_binary_character_image(
    text_pos:tuple[int, int],
    char:str,
    label:int,
    font:ImageFont.FreeTypeFont,
    resize:tuple[int, int],
    output_dir:Path,
    idx:int,
) -> np.ndarray:
    """2値化な文字画像を作成

    Args:
        text_pos (tuple[int, int]): _description_
        char (str): _description_
        label (int): _description_
        font (ImageFont.FreeTypeFont): _description_
        resize (tuple[int, int]): _description_
        output_dir (Path): _description_
        idx (int): _description_

    Returns:
        np.ndarray: _description_
    """
    # TODO: canvas_sizeの検討
    # ①②などはfont.getlengthで取得しないとキャンバスサイズが不足している
    
    # テキスト描画位置に合わせてキャンバスサイズを再調整
    canvas_size = \
        clamp(font.size + abs(text_pos[0]), 0, font.size), \
        clamp(font.size + abs(text_pos[1]), 0, font.size)
    
    # 文字描画
    text_layer = Image.new("L", canvas_size)
    text_drawer = ImageDraw.Draw(text_layer)
    text_drawer.text(text_pos, char, (255), font)
    
    # PIL to numpy
    text_layer = np.array(text_layer)
    
    text_layer = cv2.resize(text_layer, resize, interpolation=cv2.INTER_LINEAR)
    
    # binary threshold
    _, text_layer = cv2.threshold(text_layer, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    with open(str(output_dir / f"{idx}_{label}.pickle"), mode="wb") as f:
        pickle.dump(text_layer, f, -1)
    
    return text_layer
