from dataclasses import dataclass
import os
from pathlib import Path
from typing import Union
from PIL import Image, ImageFont, ImageDraw
import wget
import gzip
from contextlib import closing
import sqlite3
import numpy as np
import random
import pickle

from common import *


__all__ = [
    "WordConfig",
    "create_word_image",
]


@dataclass
class WordConfig:
    """単語画像生成の設定
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
    
    words:tuple[str, ...]
    
    def __init__(
        self,
        common:CommonConfig,
        n_train:int,
        n_valid:int,
        bg_x_params:list[int],
        bg_y_params:list[int],
        max_plots:int,
        wordnet_database_dir:str,
        en_ratio:float,
        ja_ratio:float,
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
        
        self.load_wordnet(wordnet_database_dir, en_ratio, ja_ratio)
    
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
    def random_word(self) -> str:
        """単語リストからランダムに取得

        Returns:
            str: _description_
        """
        try:
            index = self.dump_word_indexes.pop()
            return self.words[index]
        except Exception as e:
            self.dump_word_indexes = [i for i in range(len(self.words))]
            random.shuffle(self.dump_word_indexes)
            return self.random_word
    
    @property
    def random_bracket(self) -> Union[tuple[str, str], None]:
        """確率でランダムな囲い文字を取得

        Returns:
            Union[tuple[str, str], None]: _description_
        """
        return self.common.random_bracket if self.is_insert_bracket else None
    
    def setup_wordnet_database_path(
        self,
        output_dir:str,
        download_url:str="https://github.com/bond-lab/wnja/releases/download/v1.1/wnjpn.db.gz",
    ) -> Path:
        """WordNetのデータベースを取得

        Args:
            output_dir (str): 保存先
            download_url (_type_, optional): _description_. Defaults to "https://github.com/bond-lab/wnja/releases/download/v1.1/wnjpn.db.gz".

        Returns:
            Path: _description_
        """
        output_dir:Path = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        database_path = output_dir / "wnjpn.db"
        
        if not database_path.is_file():
            temp_database_path = output_dir / "wnjpn.db.gz"
            
            # download
            if not temp_database_path.is_file():
                wget.download(download_url, str(temp_database_path))
            
            # 解凍
            with gzip.open(str(temp_database_path), mode="rb") as rf:
                with open(str(database_path), mode="wb") as wf:
                    wf.write(rf.read())
            
            # 解凍出来たら*.gzipは削除
            if database_path.is_file():
                os.remove(str(temp_database_path))

        return database_path
    
    def load_wordnet(self, wordnet_dir:str, en_ratio:float, ja_ratio:float) -> None:
        """WordNetのデータベースから単語リストの読込

        Args:
            wordnet_dir (str): _description_
        """
        path = self.setup_wordnet_database_path(wordnet_dir)
        
        with closing(sqlite3.connect(path)) as conn:
            en_words = np.array(list(conn.cursor().execute('select * from word where lang="eng"')))[:, 2].tolist()
            random.shuffle(en_words)
            
            ja_words = np.array(list(conn.cursor().execute('select * from word where lang="jpn"')))[:, 2].tolist()
            random.shuffle(ja_words)
        
        en_words = en_words[0:int(len(en_words) * en_ratio)]
        ja_words = ja_words[0:int(len(ja_words) * ja_ratio)]
        
        self.words = en_words + ja_words
        random.shuffle(self.words)
    
    def _create_iter(self, n_trials:int):
        return iter([
            (
                self.common.random_background_path,
                self.bg_size,
                self.bg_crop_step,
                [(self.random_word, self.random_bracket) for i in range(self.max_plots)],
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


def create_word_image(
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
) -> None:
    """背景画像に単語をプロットした画像を作成

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
    background_layer = Image.open(background_path)
    background_layer = random_crop(background_layer, bg_size, bg_crop_step)
    background_layer = background_layer.convert("RGBA")

    text_layer = Image.new("RGBA", bg_size)
    text_drawer = ImageDraw.Draw(text_layer)
    
    char_bboxes:list[tuple[int, int, int, int]] = []
    
    bbox_map = np.zeros(bg_size[::-1], dtype=np.uint8)
    
    for text, bracket in text_and_bracket:
        if bracket is not None:
            open_bracket, close_bracket = bracket
            text = f"{open_bracket}{text}{close_bracket}"
        
        text = half2full(text)
        text = remove_unsupported_char(text, cmap)
        
        text_pos = random_text_pos(text, bg_size, margin, font, bbox_map)
        if text_pos is None:
            continue
        
        xmin, ymin, xmax, ymax = text_pos
        bbox_map[ymin:ymax, xmin:xmax] = 255
        
        text_pos = text_pos[:2]
        
        draw_text(
            text_drawer,
            font,
            text_pos,
            text,
            text_color,
            None,
            shadow,
            shadow_offset,
            shadow_color,
            outline,
            outline_width,
            outline_color,
        )
        
        calc_char_bboxes(text_pos, text, font, char_bboxes)
    
    output_layer = Image.alpha_composite(background_layer, text_layer)
    output_layer = output_layer.convert("RGB")
    output_layer.save(str(output_dir / f"image_{idx}.png"))
    
    with open(str(output_dir / f"bboxes_{idx}.pickle"), mode="wb") as f:
        pickle.dump(char_bboxes, f, -1)
