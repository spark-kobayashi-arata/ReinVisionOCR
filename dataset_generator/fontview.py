from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import yaml
import random
from dataclasses import dataclass
from typing import Union
from pathlib import Path

from common import *


@dataclass
class FontConfig:
    """フォントの設定
    """
    def __init__(
        self,
        background_dir:Union[str, list[str]],
        font_path:str,
        font_size:int,
        text_color:list[int],
        shadow:bool,
        shadow_offset:list[int],
        shadow_color:list[int],
        shadow_alpha:list[int],
        outline:bool,
        outline_width:int,
        outline_color:list[int],
        outline_alpha:list[int],
        *args,
        **kwargs,
    ):

        if isinstance(background_dir, str):
            self.background_paths = tuple([path for path in Path(background_dir).glob("*") if path.suffix in [".png", ".jpg"]])
        elif isinstance(background_dir, list):
            self.background_paths = tuple([path for child_background_dir in background_dir for path in Path(child_background_dir).glob("*") if path.suffix in [".png", ".jpg"]])
        
        self.font = ImageFont.truetype(font_path, font_size)
        
        assert is_vector(text_color, 4), "'text_color' requires list[int] type."
        self.text_color = tuple(text_color)
        
        assert isinstance(shadow, bool), "'shadow' requires bool type."
        self.shadow = shadow
        
        assert is_vector(shadow_offset, 2), "'shadow_offset' requires list[int] type."
        self.shadow_offset = tuple(shadow_offset)
        
        assert is_vector(shadow_color, 3), "'shadow_color' requires list[int] type."
        self.shadow_color = tuple(shadow_color)
        
        assert is_vector(shadow_alpha, 3), "'shadow_alpha' requires list[int] type."
        shadow_alpha[1] += 1   # random.randrange用にshadow_alpha[1]をカウントアップ
        self.shadow_alpha = tuple(shadow_alpha)
        
        assert isinstance(outline, bool), "'outline' requires bool type."
        self.outline = outline
        
        assert isinstance(outline_width, int), "'outline_width' requires int type."
        self.outline_width = outline_width
        
        assert is_vector(outline_color, 3), "'outline_color' requires list[int] type."
        self.outline_color = tuple(outline_color)
        
        assert is_vector(outline_alpha, 3), "'outline_alpha' requires list[int] type."
        outline_alpha[1] += 1   # random.randrange用にoutline_alpha[1]をカウントアップ
        self.outline_alpha = tuple(outline_alpha)
    
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
    def random_shadow_alpha(self) -> int:
        """ランダムな影色の透明度を取得

        Returns:
            int: _description_
        """
        return random.randrange(*self.shadow_alpha)
    
    @property
    def shadow_color_with_random_alpha(self) -> tuple[int, int, int, int]:
        """影色を取得、透明度はランダム

        Returns:
            tuple[int, int, int, int]: _description_
        """
        return *self.shadow_color, self.random_shadow_alpha
    
    @property
    def random_outline_alpha(self) -> int:
        """ランダムなアウトラインの透明度を取得

        Returns:
            int: _description_
        """
        return random.randrange(*self.outline_alpha)
    
    @property
    def outline_color_with_random_alpha(self) -> tuple[int, int, int, int]:
        """アウトラインカラーを取得、透明度はランダム

        Returns:
            tuple[int, int, int, int]: _description_
        """
        return *self.outline_color, self.random_outline_alpha


def my_app(
    config_path:str,
    text:str,
):
    """フォントのプレビュー

    Args:
        config_path (str): _description_
        text (str): _description_
    """
    with open(config_path, mode="r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        config = FontConfig(**config)
    
    # プレビュー用のキャンバスサイズを作成
    canvas_size = int(config.font.getlength(text) * 1.2), int(config.font.size * 3)
    text_pos = canvas_size[0]//2, canvas_size[1]//2
    
    text_layer = Image.new("RGBA", canvas_size)
    
    text_drawer = ImageDraw.Draw(text_layer)
    
    # テキスト描画
    draw_text(
        text_drawer,
        config.font,
        text_pos,
        text,
        config.text_color,
        "mm",
        config.shadow,
        config.shadow_offset,
        config.shadow_color_with_random_alpha,
        config.outline,
        config.outline_width,
        config.outline_color_with_random_alpha,
    )
    
    # 適当な背景でプレビュー
    for n in range(100):
        # 背景描画
        background_layer = Image.open(config.random_background_path)
        background_layer = random_crop(background_layer, canvas_size, (1, 1))
        background_layer = background_layer.convert("RGBA")
    
        # alpha composite
        output_canvas = Image.alpha_composite(background_layer, text_layer)
        output_canvas = output_canvas.convert("RGB")
        output_canvas = np.array(output_canvas)
        output_canvas = cv2.cvtColor(output_canvas, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("", output_canvas)
        cv2.waitKey()


if __name__ == "__main__":
    my_app(
        r"E:\ReinVisionOCR\craft\dataset_generator\config.yaml",
        "テキストも抽出に当たるからダメなんよね。",
    )
