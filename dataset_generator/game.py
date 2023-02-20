from dataclasses import dataclass
from pathlib import Path
from typing import Union
from PIL import Image, ImageDraw, ImageFont
import random
import json
import pickle

from common import *


__all__ = [
    "GameConfig",
    "create_game_image",
]


@dataclass
class GameConfig:
    common:CommonConfig
    
    n_train:int
    n_valid:int
    
    text_list:tuple[str, ...]
    
    choice_bracket:int
    
    spacing:int
    new_line_pos:int
    
    text_x_params:tuple[int, int, int]
    text_y_params:tuple[int, int, int]
    
    bg_x_params:tuple[int, int, int, int]
    bg_y_params:tuple[int, int, int, int]
    
    canvas_size:tuple[int, int]
    canvas_color:tuple[int, int, int]
    
    window_color:list[tuple[int, int, int]]
    window_alpha:tuple[int, int, int]
    
    def __init__(
        self,
        common:CommonConfig,
        n_train:int,
        n_valid:int,
        transdata_path:str,
        choice_bracket:int,
        spacing:int,
        new_line_pos:int,
        text_x_params:list[int],
        text_y_params:list[int],
        bg_x_params:list[int],
        bg_y_params:list[int],
        canvas_size:list[int],
        canvas_color:list[int],
        window_color:list[list[int]],
        window_alpha:list[int],
    ):
        self.common = common
        
        self.n_train = n_train
        self.n_valid = n_valid
        
        self.choice_bracket = choice_bracket
        
        self.spacing = spacing
        self.new_line_pos = new_line_pos
        
        assert is_vector(text_x_params, 3, int), "'text_x_params' requires list[int] type."
        text_x_params[1] += 1
        self.text_x_params = tuple(text_x_params)
        
        assert is_vector(text_y_params, 3, int), "'text_y_params' requires list[int] type."
        text_y_params[1] += 1
        self.text_y_params = tuple(text_y_params)
        
        assert is_vector(bg_x_params, 4, int), "'bg_x_params' requires list[int] type."
        bg_x_params[1] += 1
        self.bg_x_params = tuple(bg_x_params)
        
        assert is_vector(bg_y_params, 4, int), "'bg_y_params' requires list[int] type."
        bg_y_params[1] += 1
        self.bg_y_params = tuple(bg_y_params)
        
        assert is_vector(canvas_size, 2, int), "'canvas_size' requires list[int] type."
        self.canvas_size = tuple(canvas_size)
        
        assert is_vector(canvas_color, 3, int), "'canvas_color' requires list[int] type."
        self.canvas_color = tuple(canvas_color)
        
        def check_window_color(color:list[int]) -> tuple[int, int, int]:
            assert is_vector(color, 3, int), "'window_color' requires list[int] type."
            return tuple(color)
        
        self.window_color = [check_window_color(color) for color in window_color]
        
        assert is_vector(window_alpha, 3, int), "'window_alpha' requires list[int] type."
        self.window_alpha = tuple(window_alpha)
        
        self.load_transdata(transdata_path)
    
    @property
    def random_window_alpha(self) -> int:
        """ランダムなウィンドウカラーの透明度を取得

        Returns:
            int: _description_
        """
        return random.randrange(*self.window_alpha)
    
    @property
    def random_window_color(self) -> tuple[int, int, int, int]:
        """ウィンドウカラーリストからランダムに取得、透明度もランダム

        Returns:
            tuple[int, int, int, int]: _description_
        """
        try:
            index = self.dump_window_color_indexes.pop()
            return *self.window_color[index], self.random_window_alpha
        except Exception as e:
            self.dump_window_color_indexes = [i for i in range(len(self.window_color))]
            random.shuffle(self.dump_window_color_indexes)
            return self.random_window_color
    
    @property
    def random_bg_size(self) -> tuple[int, int]:
        """ランダムな背景サイズを取得

        Returns:
            tuple[int, int]: _description_
        """
        return random.randrange(*self.bg_x_params[:3]), random.randrange(*self.bg_y_params[:3])
    
    @property
    def bg_crop_step(self) -> tuple[int, int]:
        """背景の切り抜き位置?幅?間隔

        Returns:
            tuple[int, int]: _description_
        """
        return self.bg_x_params[3], self.bg_y_params[3]
    
    @property
    def random_text_pos(self) -> tuple[int, int]:
        """ランダムなテキスト描画位置を取得

        Returns:
            tuple[int, int]: _description_
        """
        return random.randrange(*self.text_x_params), random.randrange(*self.text_y_params)
    
    @property
    def is_insert_bracket(self) -> bool:
        """囲い文字を挿入するか
        
        1/choice_bracketの確率でTrueを返す

        Returns:
            bool: _description_
        """
        return bool(random.randrange(0, self.choice_bracket) == 0)
    
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
                self.random_bg_size,
                self.bg_crop_step,
                self.random_text,
                self.random_bracket,
                self.random_text_pos,
                self.spacing,
                self.new_line_pos,
                self.random_window_color,
                self.canvas_size,
                self.canvas_color,
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


def create_game_image(
    background_path:str,
    bg_size:tuple[int, int],
    bg_crop_step:tuple[int, int],
    text:str,
    bracket:Union[tuple[str, str], None],
    text_pos:tuple[int, int],
    spacing:int,
    new_line_pos:int,
    window_color:tuple[int, int, int, int],
    canvas_size:tuple[int, int],
    canvas_color:tuple[int, int, int],
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
    """ノベルゲーム風なテキスト画像を生成

    Args:
        background_path (str): _description_
        bg_size (tuple[int, int]): _description_
        bg_crop_step (tuple[int, int]): _description_
        text (str): _description_
        bracket (Union[tuple[str, str], None]): _description_
        text_pos (tuple[int, int]): _description_
        spacing (int): _description_
        new_line_pos (int): _description_
        window_color (tuple[int, int, int, int]): _description_
        canvas_size (tuple[int, int]): _description_
        canvas_color (tuple[int, int, int]): _description_
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
    # 改行時の頭に挿入する文字列
    insert_text = ""
    
    if bracket is not None:
        # 囲い文字挿入時に文末が句読点は不適切なので除外
        if text[-1] == "。" or text[-1] == "、":
            text = text[:-1]
        
        open_bracket, close_bracket = bracket
        text = f"{open_bracket}{text}{close_bracket}"
        
        # 改行時の行頭に全角スペースを挿入
        insert_text = "　"

    text = half2full(text)
    text = remove_unsupported_char(text, cmap)
    
    text_lines = split_text_lines(text, new_line_pos, insert_text)
    
    # プロットに必要な最小領域
    # テキスト描画位置のオフセットは余裕を持たせて2倍しておく
    min_bg_size = \
        text_pos[0] * 2 + max([int(font.getlength(text_line)) for text_line in text_lines]), \
        text_pos[1] * 2 + len(text_lines) * font.size + (len(text_lines) - 1) * spacing
    
    # キャンバスサイズが不足している場合は作成失敗
    assert min_bg_size[0] < canvas_size[0], "'canvas_size[0]' is too small."
    assert min_bg_size[1] < canvas_size[1], "'canvas_size[1]' is too small."
    
    # 背景サイズが不足してる場合は拡張する
    if min_bg_size[0] > bg_size[0]:
        bg_size = min_bg_size[0], bg_size[1]
    
    if min_bg_size[1] > bg_size[1]:
        bg_size = bg_size[0], min_bg_size[1]
    
    # 背景をキャンバスの中央に貼り付ける
    paste_pos = (canvas_size[0] - bg_size[0])//2, (canvas_size[1] - bg_size[1])//2, 
    
    text_layer = Image.new("RGBA", bg_size)
    text_drawer = ImageDraw.Draw(text_layer)
    
    char_bboxes:list[tuple[int, int, int, int]] = []
    
    for text_line in text_lines:
        
        draw_text(
            text_drawer,
            font,
            text_pos,
            text_line,
            text_color,
            None,
            shadow,
            shadow_offset,
            shadow_color,
            outline,
            outline_width,
            outline_color,
        )
        
        calc_char_bboxes(add_tuple(text_pos, paste_pos), text_line, font, char_bboxes)
        
        # 改行移動
        text_pos = text_pos[0], text_pos[1] + font.size + spacing
    
    background_layer = Image.open(background_path)
    background_layer = random_crop(background_layer, bg_size, bg_crop_step)
    background_layer = background_layer.convert("RGBA")
    
    window_color_layer = create_window_color_layer(bg_size, window_color)
    
    # ウィンドウカラー > 背景 > テキストの順で合成
    composite_layer = Image.alpha_composite(background_layer, window_color_layer)
    composite_layer = Image.alpha_composite(composite_layer, text_layer)
    composite_layer = composite_layer.convert("RGB")
    
    # キャンバスに貼り付ける
    output_layer = Image.new("RGB", canvas_size, canvas_color)
    output_layer.paste(composite_layer, paste_pos)
    output_layer.save(str(output_dir / f"image_{idx}.png"))
    
    with open(str(output_dir / f"bboxes_{idx}.pickle"), mode="wb") as f:
        pickle.dump(char_bboxes, f, -1)
