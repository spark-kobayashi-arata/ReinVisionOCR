from dataclasses import dataclass
import os
from pathlib import Path
from typing import Union
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import random
import pickle
from fontTools.ttLib import TTFont
import itertools

from common import *


__all__ = [
    "CharacterConfig",
    "create_character_image",
]


@dataclass
class CharacterConfig:
    max_workers:int
    
    output_dir:Path
    output_version:int
    
    background_paths:tuple[Path, ...]
    
    font:ImageFont.FreeTypeFont
    cmap:tuple[int, ...]
    
    text_color:tuple[int, int, int, int]
    
    shadow:bool
    shadow_offset:tuple[int, int]
    shadow_color:tuple[int, int, int, int]
    
    outline:bool
    outline_width:int
    outline_color:tuple[int, int, int, int]
    
    bg_x_params:tuple[int, int, int, int]
    bg_y_params:tuple[int, int, int, int]
    
    n_train:int
    n_valid:int
    
    code_list:tuple[str, ...]
    
    def __init__(
        self,
        max_workers:int,
        output_dir:str,
        background_dir:Union[str, list[str]],
        font_path:str,
        font_size:int,
        text_color:list[int],
        shadow:bool,
        shadow_offset:list[int],
        shadow_color:list[int],
        outline:bool,
        outline_width:int,
        outline_color:list[int],
        text_offset:list[int],
        bg_x_params:list[int],
        bg_y_params:list[int],
        n_valid:int,
        code_params:list[list[Union[int, str]]],
    ):
        self.max_workers = os.cpu_count() if max_workers == -1 else max_workers
        
        self.output_dir = Path(output_dir)
        self.output_version = [int(dir.stem.split("_")[1]) for dir in self.output_dir.glob("**") if dir.name.startswith("version_")]
        self.output_version = max(self.output_version) + 1 if len(self.output_version) > 0 else 0
        self.output_dir = self.output_dir / f"version_{self.output_version}"
        
        if isinstance(background_dir, str):
            self.background_paths = tuple([path for path in Path(background_dir).glob("*") if path.suffix in [".png", ".jpg"]])
        elif isinstance(background_dir, list):
            self.background_paths = tuple([path for child_background_dir in background_dir for path in Path(child_background_dir).glob("*") if path.suffix in [".png", ".jpg"]])
        
        self.font = ImageFont.truetype(font_path, font_size)
        self.cmap = tuple(TTFont(font_path).getBestCmap())
        
        assert is_vector(text_color, 4), "'text_color' requires list[int] type."
        self.text_color = tuple(text_color)
        
        assert isinstance(shadow, bool), "'shadow' requires bool type."
        self.shadow = shadow
        
        assert is_vector(shadow_offset, 2), "'shadow_offset' requires list[int] type."
        self.shadow_offset = tuple(shadow_offset)
        
        assert is_vector(shadow_color, 4), "'shadow_color' requires list[int] type."
        self.shadow_color = tuple(shadow_color)
        
        assert isinstance(outline, bool), "'outline' requires bool type."
        self.outline = outline
        
        assert isinstance(outline_width, int), "'outline_width' requires int type."
        self.outline_width = outline_width
        
        assert is_vector(outline_color, 4), "'outline_color' requires list[int] type."
        self.outline_color = tuple(outline_color)
        
        self.text_offset = [(p, p) for p in text_offset]
        self.text_offset.extend(itertools.combinations(text_offset, 2))
        
        assert is_vector(bg_x_params, 4), "'bg_x_params' requires list[int] type."
        bg_x_params[1] += 1
        self.bg_x_params = tuple(bg_x_params)
        
        assert is_vector(bg_y_params, 4), "'bg_y_params' requires list[int] type."
        bg_y_params[1] += 1
        self.bg_y_params = tuple(bg_y_params)
        
        self.n_valid = n_valid
        
        self.code_list = self.create_code_list(code_params)
    
    @property
    def random_background_path(self) -> str:
        """背景パスリストからランダムに取得

        Returns:
            str: _description_
        """
        try:
            index = self.dump_background_indexes.pop()
            return str(self.background_paths[index])
        except Exception as e:
            self.dump_background_indexes = [i for i in range(len(self.background_paths))]
            random.shuffle(self.dump_background_indexes)
            return self.random_background_path
    
    @property
    def random_bg_size(self) -> tuple[int, int]:
        """ランダムな背景サイズを取得

        Returns:
            tuple[int, int]: _description_
        """
        return random.randrange(*self.bg_x_params[:3]), random.randrange(*self.bg_y_params[:3])
    
    @property
    def bg_crop_step(self) -> tuple[int, int]:
        """背景の切り抜きステップ

        Returns:
            tuple[int, int]: _description_
        """
        return self.bg_x_params[3], self.bg_y_params[3]
    
    @property
    def random_text_offset(self) -> tuple[int, int]:
        """ランダムなテキストオフセットを取得

        Returns:
            str: _description_
        """
        try:
            index = self.dump_text_offset_indexes.pop()
            return self.text_offset[index]
        except Exception as e:
            self.dump_text_offset_indexes = [i for i in range(len(self.text_offset))]
            random.shuffle(self.dump_text_offset_indexes)
            return self.random_text_offset
    
    @property
    def font_style(self):
        """フォントの文字修飾などを一括取得

        Returns:
            _type_: _description_
        """
        return self.font \
            , self.text_color \
            , self.shadow \
            , self.shadow_offset \
            , self.shadow_color \
            , self.outline \
            , self.outline_width \
            , self.outline_color
    
    def create_code_list(self, code_params:list[list[Union[int, str]]]) -> tuple[str, ...]:
        """生成文字リストの作成

        Args:
            code_params (list[list[Union[int, str]]]): _description_

        Returns:
            tuple[str, ...]: _description_
        """
        # 変換テーブルの作成
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
        
        # 未対応文字と重複削除
        code_list = set([code for char in text if (code:=ord(char)) in self.cmap])
        code_list = sorted(code_list)
        code_list = tuple([chr(code) for code in code_list])
        return code_list
    
    def create_train_iter(self):
        return iter([
            (
                background_path,
                self.random_bg_size,
                self.bg_crop_step,
                char,
                offset,
                *self.font_style,
            )
            for background_path in self.background_paths
            for char in self.code_list
            for offset in self.text_offset
        ])
    
    def create_valid_iter(self):
        return iter([
            (
                background_path,
                self.random_bg_size,
                self.bg_crop_step,
                char,
                self.random_text_offset,
                *self.font_style,
            )
            for background_path in self.background_paths
            for char in self.code_list
            for n in range(self.n_valid)
        ])
    
    def create_iter(self, stage:StageType):
        if stage is StageType.TRAIN:
            return self.create_train_iter()
        elif stage is StageType.VALID:
            return self.create_valid_iter()
        else:
            assert False, "not support."


def create_character_image(
    background_path:str,
    bg_size:tuple[int, int],
    bg_crop_step:tuple[int, int],
    char:str,
    text_offset:tuple[int, int],
    font:ImageFont.FreeTypeFont,
    text_color:tuple[int, int, int, int],
    shadow:bool,
    shadow_offset:tuple[int, int],
    shadow_color:tuple[int, int, int, int],
    outline:bool,
    outline_width:int,
    outline_color:tuple[int, int, int, int],
    output_dir:Path,
    idx:int,
) -> np.ndarray:
    """背景に1文字プロットした画像を生成

    Args:
        background_path (str): _description_
        bg_size (tuple[int, int]): _description_
        bg_crop_step (tuple[int, int]): _description_
        char (str): _description_
        text_offset (tuple[int, int]): _description_
        font (ImageFont.FreeTypeFont): _description_
        text_color (tuple[int, int, int, int]): _description_
        shadow (bool): _description_
        shadow_offset (tuple[int, int]): _description_
        shadow_color (tuple[int, int, int, int]): _description_
        outline (bool): _description_
        outline_width (int): _description_
        outline_color (tuple[int, int, int, int]): _description_
        output_dir (Path): _description_
        idx (int): _description_
    """
    # 文字描画位置は中央寄せ
    text_pos = bg_size[0]//2 + text_offset[0], bg_size[1]//2 + text_offset[1]
    
    # アルファ付き文字
    text_layer = Image.new("RGBA", bg_size)
    text_drawer = ImageDraw.Draw(text_layer)
    
    draw_text(
        text_drawer,
        font,
        text_pos,
        char,
        text_color,
        "mm",
        shadow,
        shadow_offset,
        shadow_color,
        outline,
        outline_width,
        outline_color,
    )
    
    # グレースケールな文字
    region_layer = Image.new("L", bg_size)
    region_drawer = ImageDraw.Draw(region_layer)
    region_drawer.text(text_pos, char, (255), font, "mm")
    
    # 背景切り抜き
    background_layer = Image.open(background_path)
    background_layer = random_crop(background_layer, bg_size, bg_crop_step)
    background_layer = background_layer.convert("RGBA")

    # 背景とアルファ付き文字の合成
    image_layer = Image.alpha_composite(background_layer, text_layer)
    image_layer = image_layer.convert("RGB")
    
    # pillow to numpy
    image_layer = np.array(image_layer)
    region_layer = np.array(region_layer)
    
    # composite
    composite_layer = np.dstack([image_layer, region_layer])
    
    with open(str(output_dir / f"image_{idx}.pickle"), mode="wb") as f:
        pickle.dump(composite_layer, f)
