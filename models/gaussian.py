import numpy as np
import math
import cv2


__all__ = [
    "region_to_bboxes",
]


def region_to_bboxes(
    region:np.ndarray,
    binary_threshold:float,
    char_size:int,
    char_threshold:float,
) -> list[tuple[int, int, int, int]]:
    """regionマップからbbox抽出

    Args:
        region (np.ndarray): _description_
        binary_threshold (float): ヒートマップを2値化する際の閾値
        char_size (int): 文字領域と判定する最小面積
        char_threshold (float): 文字領域に含まれているべき最小階調の閾値

    Returns:
        list[tuple[int, int, int, int]]: _description_
    """
    _, text_score = cv2.threshold(region, binary_threshold, 1.0, cv2.THRESH_BINARY)
    text_score = np.clip(text_score * 255, 0, 255).astype(np.uint8)

    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(text_score, connectivity=4)
    
    bboxes:list[tuple[int, int, int, int]] = []
    
    for k in range(1, nlabels):
        x, y, w, h, size = stats[k].tolist()
                        
        # size filtering
        if size < char_size:
            continue
        
        # thresholding
        if np.max(region[labels==k]) < char_threshold:
            continue
        
        # make segmentation map
        segmap = np.zeros(region.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        
        # boundary check
        xmin, ymin = max(0, x - niter), max(0, y - niter)
        xmax, ymax = min(x + w + niter + 1, region.shape[1]), min(y + h + niter + 1, region.shape[0])
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[ymin:ymax, xmin:xmax] = cv2.dilate(segmap[ymin:ymax, xmin:xmax], kernel)
        
        # make box
        np_contours = np.roll(np.array(np.where(segmap!=0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box:np.ndarray = cv2.boxPoints(rectangle) * 2 # CRAFTの仕様上スケールは2で固定している
        
        # convert format
        lt, rt, lb, rb = box.tolist()
        xmin, ymin = int(min(lt[0], lb[0])), int(min(lt[1], rt[1]))
        xmax, ymax = int(max(rt[0], rb[0])), int(max(lb[1], rb[1]))
        
        bboxes.append((xmin, ymin, xmax, ymax))
    
    return bboxes
