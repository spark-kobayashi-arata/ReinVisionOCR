from pathlib import Path
import pickle
import cv2
import numpy as np

from models import *
from dataset_generator.common import *


def my_app(
    dataset_dir:str,
    dratio:float,
):
    """GaussianGeneratorの最適なパラメータ探索をする

    Args:
        dataset_dir (str): _description_
        dratio (float): _description_
    """
    image_paths = [path for path in Path(dataset_dir).glob("*.png") if path.stem.startswith("image_")]
    image_paths.sort(key=lambda path:path.stem.split("_")[1], reverse=False)
    
    bboxes_paths = [path for path in Path(dataset_dir).glob("*.pickle") if path.stem.startswith("bboxes_")]
    bboxes_paths.sort(key=lambda path:path.stem.split("_")[1], reverse=False)
    
    # 長さが異なることはあり得ないはずだけど念のため
    assert len(image_paths) == len(bboxes_paths), "not match length."
    
    # dsizeはfont_sizeより大きければ良いので基本的に指定不要
    gaussian = GaussianGenerator(dratio=dratio)

    for image_path, bboxes_path in zip(image_paths, bboxes_paths):
        image = cv2.imread(str(image_path))
        
        with open(str(bboxes_path), mode="rb") as f:
            bboxes = pickle.load(f)
        
        # 実際のregion用にリサイズ
        region_size = (image.shape[0]//2, image.shape[1]//2)
        bboxes = [(xmin//2, ymin//2, xmax//2, ymax//2) for xmin, ymin, xmax, ymax in bboxes]
        
        region = gaussian(region_size, bboxes)
        
        new_bboxes = region_to_bboxes(region, 0.01, 30, 0.1)
        for new_bbox in new_bboxes:
            cv2.rectangle(image, new_bbox[:2], new_bbox[2:], (0, 255, 0), 1)
        
        region = cv2.resize(region, image.shape[:2])
        region = cv2.cvtColor(region, cv2.COLOR_GRAY2RGB)
        
        preview = np.hstack([image, region])
        
        cv2.imshow("", preview)
        cv2.waitKey()


if __name__ == "__main__":
    my_app(
        r"E:\ReinVisionOCR\craft\dataset\version_0\valid",
        4.6,
    )
