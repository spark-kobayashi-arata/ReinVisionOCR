import yaml
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Union
from fontTools.ttLib import TTFont
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
import random

from common import *


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
    
    code_list:tuple[str, ...]
    
    resize:tuple[int, int]
    
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
        bg_x_params:list[int],
        bg_y_params:list[int],
        code_params:list[list[Union[int, str]]],
        resize:list[int],
        *args,
        **kwargs,
    ):
        self.max_workers = os.cpu_count() if max_workers == -1 else max_workers
        
        # プレビューの結果は一時的な利用なので上書きを許可
        self.output_dir = Path(output_dir)
        
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
        
        assert is_vector(bg_x_params, 4), "'bg_x_params' requires list[int] type."
        bg_x_params[1] += 1
        self.bg_x_params = tuple(bg_x_params)
        
        assert is_vector(bg_y_params, 4), "'bg_y_params' requires list[int] type."
        bg_y_params[1] += 1
        self.bg_y_params = tuple(bg_y_params)
        
        assert is_vector(resize, 2), "'resize' requires list[int] type."
        self.resize = resize
        
        self.code_list = self.create_code_list(code_params)
    
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
    
    def create_iter(self):
        return iter([
            (
                self.background_paths[0],
                self.random_bg_size,
                self.bg_crop_step,
                char,
                *self.font_style,
                self.resize,
            )
            for char in self.code_list
        ])


def create_character_image(
    background_path:str,
    bg_size:tuple[int, int],
    bg_crop_step:tuple[int, int],
    char:str,
    font:ImageFont.FreeTypeFont,
    text_color:tuple[int, int, int, int],
    shadow:bool,
    shadow_offset:tuple[int, int],
    shadow_color:tuple[int, int, int, int],
    outline:bool,
    outline_width:int,
    outline_color:tuple[int, int, int, int],
    resize:tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
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
    """
    # 文字描画位置は中央寄せ
    text_pos = bg_size[0]//2, bg_size[1]//2
    
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
    
    # resize
    image_layer = cv2.resize(image_layer, resize, interpolation=cv2.INTER_LINEAR)
    region_layer = cv2.resize(region_layer, resize, interpolation=cv2.INTER_LINEAR)
    
    # format
    image_layer = cv2.cvtColor(image_layer, cv2.COLOR_RGB2BGR)    
    region_layer = cv2.cvtColor(region_layer, cv2.COLOR_GRAY2BGR)
    
    return image_layer, region_layer


def my_app(config_path:str, resize:int=64):
    with open(config_path, mode="r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        config = CharacterConfig(**config | {"resize": [resize, resize]})

    with ThreadPoolExecutor(config.max_workers) as executor:
        # 出力先を作成
        stage_dir = config.output_dir / "debug"
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        futures = [
            executor.submit(
                create_character_image,
                *params,
            )
            for params in config.create_iter()
        ]
        
        with TileImageGenerator(64, 64, stage_dir, "image") as tig_image, \
            TileImageGenerator(64, 64, stage_dir, "region") as tig_region:
            for future in tqdm(futures):
                image, region = future.result()
                tig_image.add(image)
                tig_region.add(region)


if __name__ == "__main__":
    my_app(r"E:\ReinVisionOCR\post-craft\dataset_generator\config.yaml")
