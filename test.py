import numpy as np
import cv2
from pathlib import Path
import torch
from torch import Tensor
from torchvision import transforms

from train import CRAFTModule
from models import region_to_bboxes


# 対応している入力サイズ
INPUT_SIZE = (224, 864, 1024)

MEAN_NORM = (0.485, 0.456, 0.406)
STD_NORM = (0.229, 0.224, 0.225)

IMAGE_SUFFIX = tuple([".png", ".jpg"])


def create_input(
    image:np.ndarray,
    transform:transforms.Compose,
) -> tuple[Tensor, tuple[int, int, int, int]]:
    """入力画像の作成

    Args:
        image (np.ndarray): _description_
        transform (transforms.Compose): _description_

    Returns:
        tuple[Tensor, tuple[int, int, int, int]]: _description_
    """
    # キャンバスサイズの推定
    image_h, image_w = image.shape[:2]
    canvas_h = min([size for size in INPUT_SIZE if image_h <= size])
    canvas_w = min([size for size in INPUT_SIZE if image_w <= size])
    
    # 貼り付け位置
    paste_x = (canvas_w - image_w)//2
    paste_y = (canvas_h - image_h)//2
    
    # キャンバスに画像を貼り付け
    input = np.full((canvas_h, canvas_w, 3), 128, dtype=np.uint8)
    input[paste_y:paste_y+image_h, paste_x:paste_x+image_w] = image
    
    input = transform(input)
    input = input.unsqueeze(0)
    input = input.cuda()
    
    return input, (paste_x, paste_y, image_w, canvas_h)


@torch.no_grad()
def my_app(
    checkpoint_path:str,
    image_dir:str,
    binary_threshold:float=0.01,
    char_size:int=30,
    size_thresh:float=0.1,
):
    image_paths = [path for path in Path(image_dir).glob("*") if path.suffix in IMAGE_SUFFIX]
    def custom_sort(path:Path):
        try:
            return int(path.stem)
        except Exception as e:
            return -1
    image_paths.sort(key=lambda path: custom_sort(path))
    
    craft = CRAFTModule.load_from_checkpoint(checkpoint_path)
    craft.eval()
    craft.cuda()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_NORM, std=STD_NORM),
    ])
    
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        input, (x, y, w, h) = create_input(image, transform)
        
        region = craft(input)
        region = region.squeeze().cpu().detach().numpy()
        
        bboxes = region_to_bboxes(region, binary_threshold, char_size, size_thresh)
        bboxes = [(max(0, xmin-x), max(0, ymin-y), min(xmax-x, w), min(ymax-y, h)) for xmin, ymin, xmax, ymax in bboxes]
    
        for idx, bbox in enumerate(bboxes):
            cv2.rectangle(image, bbox[:2], bbox[2:], (0, 255, 0), 1)
            cv2.putText(image, str(idx), (bbox[0]+1, bbox[1]+1), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(idx), bbox[:2], cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("", image)
        cv2.waitKey()


if __name__ == "__main__":
    my_app(
        r"E:\ReinVisionOCR\craft\log_logs\version_0\checkpoints\last.ckpt",
        r"E:\ReinVisionOCR\craft\dataset\version_0\valid",
    )
