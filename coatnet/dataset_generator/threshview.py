import yaml
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Union
from fontTools.ttLib import TTFont
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2

from common import *


@dataclass
class BinaryCharacterConfig:
    """2値化文字生成の設定
    """
    max_workers:int
    
    output_dir:Path
    output_version:int
    
    font_list:tuple[ImageFont.FreeTypeFont, ...]
    cmap:tuple[int, ...]
    
    code_list:tuple[str, ...]
    
    def __init__(
        self,
        max_workers:int,
        output_dir:str,
        font_path:str,
        font_size:list[int],
        resize:list[int],
        code_params:list[list[Union[int, str]]],
        *args,
        **kwargs,
    ):
        self.max_workers = os.cpu_count() if max_workers == -1 else max_workers
        
        # プレビューの結果は一時的な利用なので上書きを許可
        self.output_dir = Path(output_dir)
        
        self.font_list = tuple([ImageFont.truetype(font_path, size) for size in font_size])
        self.cmap = tuple(TTFont(font_path).getBestCmap())
        
        assert is_vector(resize, 2), "'resize' requires list[int] type."
        self.resize = tuple(resize)
        
        self.code_list = self.create_code_list(code_params)
    
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
                code,
                font,
                self.resize,
            )
            for code in self.code_list
            for font in self.font_list
        ])
    
    def create_iter2(self):
        return iter([
            (
                code,
                font,
                self.resize,
            )
            for font in self.font_list
            for code in self.code_list
        ])


def create_binary_character_image(
    char:str,
    font:ImageFont.FreeTypeFont,
    resize:tuple[int, int],
) -> np.ndarray:
    """2値化文字画像を作成

    Args:
        char (str): _description_
        font (ImageFont.FreeTypeFont): _description_
        resize (tuple[int, int]): _description_

    Returns:
        np.ndarray: _description_
    """
    canvas_size = font.size, font.size
    text_pos = font.size//2, font.size//2
    
    # 文字描画
    text_layer = Image.new("L", canvas_size)
    text_drawer = ImageDraw.Draw(text_layer)
    text_drawer.text(text_pos, char, (255), font, anchor="mm")
    
    # PIL to numpy
    text_layer = np.array(text_layer)
    
    # resize
    text_layer = cv2.resize(text_layer, resize, interpolation=cv2.INTER_LINEAR)
        
    # binary threshold
    _, text_layer = cv2.threshold(text_layer, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # grayscale to bgr
    text_layer = cv2.cvtColor(text_layer, cv2.COLOR_GRAY2BGR)
    
    return text_layer


def my_app(config_path:str):
    """2値化の結果をプレビュー

    Args:
        config_path (str): _description_
    """
    with open(config_path, mode="r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        config = BinaryCharacterConfig(**config)
    
    # NOTE: 類似文字見つける用
    debug_code_list = "".join(config.code_list)
    
    with ProcessPoolExecutor(config.max_workers) as executor:
        # 出力先を作成
        stage_dir = config.output_dir / "debug"
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        futures = [
            executor.submit(
                create_binary_character_image,
                *params,
            )
            for params in config.create_iter()
        ]
        
        with TileImageGenerator(64, 64, stage_dir, "thresh") as tig:
            for future in tqdm(futures):
                tig.add(future.result())
        
        futures = [
            executor.submit(
                create_binary_character_image,
                *params,
            )
            for params in config.create_iter2()
        ]
        
        with TileImageGenerator(64, 64, stage_dir, "similar") as tig:
            for future in tqdm(futures):
                tig.add(future.result())


if __name__ == "__main__":
    my_app(r"E:\ReinVisionOCR\coatnet\dataset_generator\config.yaml")
