from pathlib import Path
import numpy as np
import cv2
import json
from itertools import zip_longest

from rein_vision_ocr import *


IMAGE_SUFFIX = tuple([".png", ".jpg"])


class SequenceMatcher:
    def __init__(
        self,
        similars:list[list[str]],
    ):
        # 類似文字一覧を取得
        self.similar_chars = similars
        
        # 各類似文字グループの先頭
        self.label_ignores:str = "".join([similar[0] for similar in self.similar_chars])
    
    def is_similar(self, label_char:str, text_char:str) -> bool:
        
        # NOTE: ゲシュタルトパターンマッチングな都合で類似文字以外でもTrue通る可能性あるけど低いので無視している
        for similars in self.similar_chars:
            if label_char in similars and text_char in similars:
                return True
        return False
    
    def equals(self, label:str, input:str, reverse:bool) -> bool:
        l = label[0] if label is not None and len(label) > 0 else None
        i = input[0] if input is not None and len(input) > 0 else None
        
        if l == i:
            return True
        elif l is not None and self.is_similar(l, i):
            return True
        elif l is not None and i is not None:
            if reverse:
                return self.equals(label[1:], input, reverse)
            else:
                return self.equals(label, input[1:], reverse)
        
        return False
    
    def __call__(self, label:str, input:str, reverse:bool) -> tuple[list[bool], float]:
        results = [self.equals(label[i:], input[i:], reverse) for i in range(len(label))]
        return results, sum(results) / len(label)


def my_app(
    dataset_dir:str,
    similars:list[list[str]],
    craft_pretrained_path:str,
    post_craft_pretrained_path:str,
    coatnet_pretrained_path:str,
):
    # str to Path
    dataset_dir:Path = Path(dataset_dir)
    
    # 画像読込
    image_paths = [path for path in (dataset_dir / "image").glob("*") if path.suffix in IMAGE_SUFFIX]
    image_paths.sort(key=lambda path:int(path.stem))
    
    # テキスト読込
    with open(dataset_dir / "text.txt", mode="r", encoding="utf-8") as f:
        text_list = f.read()
        text_list = text_list.split("\n")
        text_list = text_list[:-1]
    
    # サイズは一致している
    assert len(image_paths) == len(text_list), "not match length."
    
    # ゲシュタルトパターンマッチングに近い精度算出方法
    diff = SequenceMatcher(similars)
    
    # ReinVisionOCR
    app = ReinVisionOCR(
        craft_pretrained_path,
        post_craft_pretrained_path,
        coatnet_pretrained_path,
    )
    
    # 合計精度
    total_ratio = 0.0
    total_count, total_count_num = 0, 0
    total_num = len(text_list)
    
    for label_idx, (image_path, label) in enumerate(zip(image_paths, text_list)):
        # 画像読込
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # run app
        text, bboxes, chars = app.image_to_text(image)
        
        # 精度算出
        results, ratio = diff(label, text, False)
        if ratio != 1.0:
            temp_results, temp_ratio = diff(label, text, True)
            if ratio < temp_ratio:
                ratio = temp_ratio
                results = temp_results
        
        # 精度カウント
        total_ratio += ratio
        total_count += sum(results)
        total_count_num += len(results)
        
        # 失敗した場合にのみプレビュー
        if ratio == 1.0:
            continue
        
        # 精度中間結果
        print(f"{label_idx:4d}/{total_num:4d} | {ratio*100:03.0f} | {text}")
        
        # 文字領域の可視化とナンバリング
        for idx, (bbox, success) in enumerate(zip_longest(bboxes, results)):
            if bbox is not None:
                cv2.rectangle(image, bbox[:2], bbox[2:], (0, 255, 0) if success is not None and success else (255, 0, 0), 1)
                cv2.putText(image, str(idx), (bbox[0]+1, bbox[1]+1), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(idx), bbox[:2], cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        
        # ピクセル抽出の可視化
        text_image = np.full_like(image, (0, 0, 0), np.uint8)
        for (xmin, ymin, xmax, ymax), char_image in zip(bboxes, chars):
            char_image = cv2.resize(char_image, (xmax-xmin, ymax-ymin), interpolation=cv2.INTER_LINEAR)
            char_image = cv2.cvtColor(char_image, cv2.COLOR_GRAY2BGR)
            text_image[ymin:ymax, xmin:xmax] = char_image
        
        # RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        composite_image = np.vstack([image, text_image])
        
        # show
        cv2.imshow("", composite_image)
        cv2.waitKey(1)
    
    # 精度結果
    print(f"{total_ratio/total_num*100:.8f}% ({total_count}/{total_count_num})")


if __name__ == "__main__":
    # 類似文字
    similars = [
        ["ロ", "口"],
        ["ー", "一"],
        ["カ", "力"],
        ["エ", "工"],
        ["べ", "ベ"],
        ["ぺ", "ペ"],
        ["０", "Ｏ"],
        ["ニ", "二"],
    ]
    
    my_app(
        r"E:\ReinVisionOCR\resources\data\test",
        similars,
        r"E:\ReinVisionOCR\resources\pretrained\msgothic\craft_v0.pth",
        r"E:\ReinVisionOCR\resources\pretrained\msgothic\post_craft_v0.pth",
        r"E:\ReinVisionOCR\resources\pretrained\msgothic\coatnet_v0.pth",
    )
