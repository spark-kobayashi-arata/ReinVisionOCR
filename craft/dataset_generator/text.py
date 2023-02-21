from dataclasses import dataclass
from pathlib import Path
from typing import Union
from PIL import ImageFont
import random
import json

from common import *
from word import *


__all__ = [
    "TextConfig",
    "create_text_image",
]


@dataclass
class TextConfig:
    """文章画像生成の設定
    """
    common:CommonConfig
    
    n_train:int
    n_valid:int
    
    bg_x_params:tuple[int, int]
    bg_y_params:tuple[int, int]
    
    max_plots:int
    
    choice_bracket:int
    
    margin_x_params:tuple[int, int, int]
    margin_y_params:tuple[int, int, int]
    
    text_list:tuple[str, ...]
    
    def __init__(
        self,
        common:CommonConfig,
        n_train:int,
        n_valid:int,
        bg_x_params:list[int],
        bg_y_params:list[int],
        max_plots:int,
        transdata_path:str,
        choice_bracket:int,
        margin_x_params:list[int],
        margin_y_params:list[int],
    ):
        self.common = common
        
        self.n_train = n_train
        self.n_valid = n_valid
        
        assert is_vector(bg_x_params, 2, int), "'bg_x_params' requires list[int] type."
        self.bg_x_params = tuple(bg_x_params)
        
        assert is_vector(bg_y_params, 2, int), "'bg_y_params' requires list[int] type."
        self.bg_y_params = tuple(bg_y_params)
        
        self.max_plots = max_plots
        
        self.choice_bracket = choice_bracket
        
        assert is_vector(margin_x_params, 3, int), "'margin_x_params' requires list[int] type."
        margin_x_params[1] += 1
        self.margin_x_params = tuple(margin_x_params)
        
        assert is_vector(margin_y_params, 3, int), "'margin_y_params' requires list[int] type."
        margin_y_params[1] += 1
        self.margin_y_params = tuple(margin_y_params)
        
        self.load_transdata(transdata_path)
    
    @property
    def bg_size(self) -> tuple[int, int]:
        """背景サイズ

        Returns:
            tuple[int, int]: _description_
        """
        return self.bg_x_params[0], self.bg_y_params[0]
    
    @property
    def bg_crop_step(self) -> tuple[int, int]:
        """背景の切り抜き位置?幅?間隔

        Returns:
            tuple[int, int]: _description_
        """
        return self.bg_x_params[1], self.bg_y_params[1]
    
    @property
    def is_insert_bracket(self) -> bool:
        """囲い文字を挿入するか
        
        1/choice_bracketの確率でTrueを返す

        Returns:
            bool: _description_
        """
        return bool(random.randrange(0, self.choice_bracket) == 0)
    
    @property
    def margin(self) -> tuple[int, int]:
        """文字周辺の余白

        Returns:
            tuple[int, int]: _description_
        """
        return random.randrange(*self.margin_x_params), random.randrange(*self.margin_y_params)
    
    @property
    def random_text(self) -> str:
        """文章リストからランダムに取得

        Returns:
            str: _description_
        """
        try:
            index = self.dump_text_indexes.pop()
            return self.text_list[index]
        except Exception as e:
            self.dump_text_indexes = [i for i in range(len(self.text_list))]
            random.shuffle(self.dump_text_indexes)
            return self.random_text
    
    @property
    def random_bracket(self) -> Union[tuple[str, str], None]:
        """確率でランダムな囲い文字を取得

        Returns:
            Union[tuple[str, str], None]: _description_
        """
        return self.common.random_bracket if self.is_insert_bracket else None
    
    def load_transdata(self, path:str) -> None:
        """*.transdataからテキストデータを抽出する

        Args:
            path (str): _description_
        """
        with open(path, mode="r", encoding="utf-8") as f:
            self.text_list = [value["text"] for value in json.load(f).values()]
    
    def _create_iter(self, n_trials:int):
        return iter([
            (
                self.common.random_background_path,
                self.bg_size,
                self.bg_crop_step,
                [(self.random_text, self.random_bracket) for i in range(self.max_plots)],
                self.margin,
            )
            for n in range(n_trials)
        ])
    
    def create_iter(self, stage:StageType):
        if stage is StageType.TRAIN:
            return self._create_iter(self.n_train)
        elif stage is StageType.VALID:
            return self._create_iter(self.n_valid)
        else:
            assert False, "not support."


def create_text_image(
    background_path:str,
    bg_size:tuple[int, int],
    bg_crop_step:tuple[int, int],
    text_and_bracket:tuple[str, Union[tuple[str, str], None]],
    margin:tuple[int, int],
    font:ImageFont.FreeTypeFont,
    cmap:tuple[int, ...],
    text_color:tuple[int, int, int, int],
    shadow:bool,
    shadow_offset:tuple[int, int],
    shadow_color:tuple[int, int, int, int],
    outline:bool,
    outline_width:int,
    outline_color:tuple[int, int, int, int],
    output_dir:Path,
    idx:int,
):
    """背景画像に文章をプロットした画像を作成

    Args:
        background_path (str): _description_
        bg_size (tuple[int, int]): _description_
        bg_crop_step (tuple[int, int]): _description_
        text_and_bracket (tuple[str, Union[tuple[str, str], None]]): _description_
        margin (tuple[int, int]): _description_
        font (ImageFont.FreeTypeFont): _description_
        cmap (tuple[int, ...]): _description_
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
    create_word_image(**locals())
