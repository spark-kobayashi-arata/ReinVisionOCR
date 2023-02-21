import os
import random
from enum import Enum, auto
from dataclasses import dataclass
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw
from typing import Union, TypeVar
from fontTools.ttLib import TTFont
import numpy as np
import cv2


__all__ = [
    "HALF2FULL",
    "StageType",
    "CommonConfig",
    "is_vector",
    "random_crop",
    "random_text_pos",
    "add_tuple",
    "calc_char_bboxes",
    "half2full",
    "remove_unsupported_char",
    "split_text_lines",
    "lerp",
    "clamp",
    "create_window_color_layer",
    "TileImageGenerator",
    "draw_text",
]


# 半角から全角の変換マップ
HALF2FULL = str.maketrans({chr(0x0021 + i): chr(0xFF01 + i) for i in range(94)})

T = TypeVar("T", int, float)


class StageType(Enum):
    TRAIN = 0
    VALID = auto()
    
    def __str__(object) -> str:
        if object is StageType.TRAIN:
            return "train"
        elif object is StageType.VALID:
            return "valid"
        else:
            raise NotImplementedError("'__str__' is not support.")


class TileImageGenerator:
    """タイル画像生成
    
    使い方
    
    with TileImageGenerator(...) as tig:
        tig.add(...)
    """

    def __init__(self, x_num:int, y_num:int, output_dir:Path, filename:str):
        self.x_num = x_num
        self.y_num = y_num
        self.x_count = 0
        self.y_count = 0
        self.col = None
        self.row = None
        self.tile_count = 0
        self.output_dir = output_dir
        self.filename = filename
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.save()

    def add(self, image:np.ndarray):
        """画像を追加

        Args:
            image (np.ndarray): _description_
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        self.__add_col(image)

    def __add_col(self, image:np.ndarray):
        self.x_count += 1
        
        if self.col is None:
            self.col = image
        else:
            sep = np.full([self.col.shape[0], 1, 3], (0,0,255), self.col.dtype)
            self.col = np.hstack([self.col, sep, image])
        
        if self.x_count >= self.x_num:
            self.add_row()
            self.col = None
            self.x_count = 0

    def add_row(self):
        self.y_count += 1
        
        if self.row is None:
            self.row = self.col
        else:
            sep = np.full([1, self.row.shape[1], 3], (0,0,255), self.col.dtype)
            self.row = np.vstack([self.row, sep, self.col])
            sep = None
        
        if self.y_count >= self.y_num:
            self.save_tile()
            self.row = None
            self.y_count = 0

    def save_tile(self):
        cv2.imwrite(str(self.output_dir / f"{self.filename}_{self.tile_count}.png"), self.row)
        self.tile_count += 1

    def save(self):
        image = None
        
        if self.col is not None:
            if self.row is None:
                image = self.col
            else:
                col = np.zeros((self.col.shape[0], self.row.shape[1] - self.col.shape[1], 3), self.row.dtype)
                col = np.hstack([self.col, col])
                image = np.vstack([self.row, col])
        elif self.row is not None:
            image = self.row
        
        if image is not None:
            cv2.imwrite(str(self.output_dir / f"{self.filename}_{self.tile_count}.png"), image)
    
    def clear(self):
        self.x_count = 0
        self.y_count = 0
        self.col = None
        self.row = None
        self.tile_count = 0


@dataclass
class CommonConfig:
    """共通設定
    """
    
    max_workers:int
    
    output_dir:Path
    output_version:int
    
    background_paths:tuple[Path, ...]
    
    font:ImageFont.FreeTypeFont
    cmap:tuple[int, ...]
    
    text_color:tuple[int, int, int, int]
    
    shadow:bool
    shadow_offset:tuple[int, int]
    shadow_color:tuple[int, int, int]
    shadow_alpha:tuple[int, int, int]
    
    outline:bool
    outline_width:int
    outline_color:tuple[int, int, int]
    outline_alpha:tuple[int, int, int]
    
    brackets:list[tuple[str, str]]
    
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
        shadow_alpha:list[int],
        outline:bool,
        outline_width:int,
        outline_color:list[int],
        outline_alpha:list[int],
        brackets:list[list[str]],
        *args,
        **kwargs,
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
        
        def to_bracket(bracket:list[str]) -> tuple[str, str]:
            """囲い文字が正しいフォーマットか判定

            Args:
                bracket (list[str]): _description_

            Returns:
                tuple[str, str]: _description_
            """
            assert is_vector(bracket, 2, str), "'bracket' requires list[str] type."
            return tuple(bracket)
        
        self.brackets = [to_bracket(bracket) for bracket in brackets]
        
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
    def random_bracket(self) -> tuple[str, str]:
        """囲い文字リストからランダムに取得

        Returns:
            tuple[str, str]: _description_
        """
        try:
            index = self.dump_bracket_indexes.pop()
            return self.brackets[index]
        except Exception as e:
            self.dump_bracket_indexes = [i for i in range(len(self.brackets))]
            random.shuffle(self.dump_bracket_indexes)
            return self.random_bracket
    
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
    
    @property
    def font_style(self):
        """フォントの文字修飾などを一括取得

        Returns:
            _type_: _description_
        """
        return self.font \
            , self.cmap \
            , self.text_color \
            , self.shadow \
            , self.shadow_offset \
            , self.shadow_color_with_random_alpha \
            , self.outline \
            , self.outline_width \
            , self.outline_color_with_random_alpha


def is_vector(
    values:Union[list[int], tuple[int, ...]],
    length:int,
    value_type:type=int,
) -> bool:
    """valuesがvalue_typeのみが含まれるvectorなフォーマットかを判定

    Args:
        values (Union[list[int], tuple[int, ...]]): _description_
        length (int): _description_
        value_type (type, optional): _description_. Defaults to int.

    Returns:
        bool: _description_
    """
    return (isinstance(values, list) or isinstance(values, tuple)) and \
        len([0 for value in values if isinstance(value, value_type)]) == length


def random_crop(
    image:Image.Image,
    crop_size:tuple[int, int],
    crop_step:tuple[int, int],
) -> Image.Image:
    """画像のランダム切り抜き

    Args:
        image (Image.Image): _description_
        crop_size (tuple[int]): 切り抜きサイズ
        crop_step (tuple[int]): _description_

    Returns:
        Image.Image: _description_
    """
    image_w, image_h = image.size
    crop_w, crop_h = crop_size

    # resize if smaller than crop size
    if image_w < crop_w or image_h < crop_h:
        wh = crop_w - image_w if image_w < crop_w else crop_h - image_h
        image = image.resize((image_w+wh, image_h+wh), resample=Image.Resampling.BILINEAR)
        image_w, image_h = image.size

    # set crop point
    x_range, y_range = image_w - crop_w, image_h - crop_h
    x, y = random.randrange(0, x_range+1, crop_step[0]), random.randrange(0, y_range+1, crop_step[1])

    return image.crop((x, y, x+crop_w, y+crop_h))


def random_text_pos(
    text:str,
    bg_size:tuple[int, int],
    margin_size:tuple[int, int],
    font:ImageFont.FreeTypeFont,
    bbox_map:np.ndarray,
    n_trials:int=64,
) -> tuple[int, int]:
    """文字描画位置を空き領域からランダムに算出

    Args:
        text (str): _description_
        bg_size (tuple[int]): _description_
        margin_size (tuple[int]): _description_
        font (ImageFont.FreeTypeFont): _description_
        bbox_map (np.ndarray): 描画済みの位置が分かるマップ
        n_trials (int, optional): 最大探索回数. Defaults to 64.

    Returns:
        tuple[int]: 空きが見つからない場合は空のtupleを返す
    """
    text_w, text_h = int(font.getlength(text)), font.size
    size_w, size_h = bg_size
    margin_w, margin_h = margin_size
            
    # テキストがキャンバスより大きい場合は描画不可
    if text_w + margin_w * 2 >= size_w:
        return None
    if text_h + margin_h * 2 >= size_h:
        return None
            
    x_range = size_w - text_w - margin_w * 2
    y_range = size_h - text_h - margin_h * 2
            
    for i in range(n_trials):
        xmin = random.randrange(0, x_range, 1) + margin_w
        ymin = random.randrange(0, y_range, 1) + margin_h
        xmax = xmin + text_w + margin_w
        ymax = ymin + text_h + margin_h
        
        if np.sum(bbox_map[ymin:ymax, xmin:xmax]) == 0:
            return (xmin, ymin, xmax, ymax)
    
    return None


def add_tuple(a:tuple[int, ...], b:tuple[int, ...]) -> tuple[int, ...]:
    """tuple同士の加算

    Args:
        a (tuple[int, ...]): _description_
        b (tuple[int, ...]): _description_

    Returns:
        tuple[int, ...]: _description_
    """
    assert len(a) == len(b), "Length of values does not match length of index."
    return tuple([x + y for x, y in zip(a, b)])


def calc_char_bboxes(
    text_pos:tuple[int, int],
    text:str,
    font:ImageFont.FreeTypeFont,
    bboxes:list[int, int, int, int],
) -> None:
    """文字単位でbbox計算
    
    Args:
        text_pos (tuple[int]): _description_
        text (str): _description_
        font (ImageFont.FreeTypeFont): _description_
        bboxes (list[int]): bboxの格納先
    """
    char_x, char_y = text_pos
    char_h = font.size
    
    for char in text:
        char_w = int(font.getlength(char))
        
        # NOTE: 有効ピクセルに偏りがある場合はbboxも寄せた方がいいのかな
        #if char in "「『":
        #    bbox = char_x + 2, char_y - 2, char_x + 2 + char_w, char_y - 2 + char_h
        #    bboxes.append(bbox)
        #elif char in "」』、。":
        #    bbox = char_x - 2, char_y + 2, char_x - 2 + char_w, char_y + 2 + char_h
        #    bboxes.append(bbox)
        #elif char in "【［（":
        #    bbox = char_x + 2, char_y, char_x + 2 + char_w, char_y + char_h
        #    bboxes.append(bbox)
        #elif char in "】］）":
        #    bbox = char_x - 2, char_y, char_x - 2 + char_w, char_y + char_h
        #    bboxes.append(bbox)
        if char != "　":
            bbox = char_x, char_y, char_x + char_w, char_y + char_h
            bboxes.append(bbox)
        
        char_x += char_w


def half2full(text:str) -> str:
    """半角 to 全角
    
    Args:
        text (str): _description_
        
    Returns:
        str: _description_
    """
    return text.translate(HALF2FULL)


def remove_unsupported_char(text:str, cmap:tuple[int, ...]) -> str:
    """非対応文字の削除
    
    Args:
        text (str): _description_
        
    Returns:
        str: _description_
    """
    if len(remove_table:={char:None for char in text if ord(char) not in cmap}) > 0:
        return text.translate(remove_table)
    return text


def split_text_lines(text:str, new_line_pos:int, insert_text:str) -> list[str]:
    """テキストを行ごとに分割して返す

    Args:
        text (str): _description_
        new_line_pos (int): 改行位置
        insert_text (str): 改行後、先頭に挿入する文字

        Returns:
        list[str]: _description_
    """
    text_lines:list[str] = []
    text_line = ""
    counter = 0
    
    for char in text:
        if counter >= new_line_pos:
            text_lines.append(text_line)
            text_line = insert_text
            counter = len(insert_text)
        text_line += char
        counter += 1
    
    text_lines.append(text_line)
      
    return text_lines


def lerp(a:T, b:T, alpha:float) -> T:
    """線形補間
    
    Args:
        a (T): _description_
        b (T): _description_
        alpha (float): _description_
    
    Returns:
        T: _description_
    """
    return (1 - alpha) * a + alpha * b


def clamp(x:T, a:T, b:T) -> T:
    """clamp

    Args:
        x (T): _description_
        a (T): _description_
        b (T): _description_

    Returns:
        T: _description_
    """
    return min(max(x, a), b)


def create_window_color_layer(
    size:tuple[int, int],
    window_color:tuple[int, int, int, int],
) -> Image.Image:
    """ウィンドウカラーレイヤーの生成
    
    Args:
        size (tuple[int]): _description_
        
    Returns:
        Image.Image: _description_
    """
    w, h = size
    color, alpha = window_color[:3], window_color[3]
    
    alpha_pixels = [int(lerp(0, alpha, y/h)) for y in range(h)]
    alpha_pixels = np.tile(alpha_pixels, (w, 1)).astype(np.uint8)
    alpha_pixels = np.transpose(alpha_pixels, (1, 0))
    color_pixels = np.full([h,w,3], color, dtype=np.uint8)
    pixels = np.dstack([color_pixels, alpha_pixels])
    return Image.fromarray(pixels)


def draw_text(
    drawer:ImageDraw.ImageDraw,
    font:ImageFont.FreeTypeFont,
    text_pos:tuple[int, int],
    text:str,
    text_color:tuple[int, int, int, int],
    shadow:bool=False,
    shadow_offset:Union[tuple[int, int], None]=None,
    shadow_color:Union[tuple[int, int, int, int], None]=None,
    outline:bool=False,
    outline_width:Union[int, None]=None,
    outline_color:Union[tuple[int, int, int, int], None]=None,
) -> None:
    """テキスト描画

    Args:
        drawer (ImageDraw.ImageDraw): _description_
        font (ImageFont.FreeTypeFont): _description_
        text_pos (tuple[int, int]): _description_
        text (str): _description_
        text_color (tuple[int, int, int, int]): _description_
        shadow (bool, optional): _description_. Defaults to False.
        shadow_offset (Union[tuple[int, int], None], optional): _description_. Defaults to None.
        shadow_color (Union[tuple[int, int, int, int], None], optional): _description_. Defaults to None.
        outline (bool, optional): _description_. Defaults to False.
        outline_width (Union[int, None], optional): _description_. Defaults to None.
        outline_color (Union[tuple[int, int, int, int], None], optional): _description_. Defaults to None.
    """
    if shadow and outline:
        drawer.text(add_tuple(text_pos, shadow_offset), text, shadow_color, font, stroke_width=outline_width, stroke_fill=shadow_color)
    elif shadow:
        drawer.text(add_tuple(text_pos, shadow_offset), text, shadow_color, font)

    if outline:
        drawer.text(text_pos, text, text_color, font, stroke_width=outline_width, stroke_fill=outline_color)
    else:
        drawer.text(text_pos, text, text_color, font)
