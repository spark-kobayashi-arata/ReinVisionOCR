import numpy as np
import math
import cv2


__all__ = [
    "region_to_bboxes",
    "GaussianGenerator",
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
        list[tuple[int]]: _description_
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
        lt, rt, lb, rb = box.astype(np.int32).tolist()
        xmin, ymin = min(lt[0], lb[0]), min(lt[1], rt[1])
        xmax, ymax = max(rt[0], rb[0]), max(lb[1], rb[1])
        
        bboxes.append((xmin, ymin, xmax, ymax))
    
    return bboxes


class GaussianGenerator:
    def __init__(
        self,
        ksize:tuple[int, int]=(64, 64),
        dratio:float=5.0,
    ):
        self.ksize = ksize
        self.dratio = dratio
        self.gaussian2d = self.isotropic_gaussian_heatmap(ksize=ksize, dratio=dratio)
    
    def isotropic_gaussian_heatmap(
        self,
        ksize:tuple[int, int]=(32, 32),
        dratio:float=3.0,
    ) -> np.ndarray:
        """_summary_

        Args:
            ksize (tuple[int], optional): _description_. Defaults to (32, 32).
            dratio (float, optional): _description_. Defaults to 3.0.

        Returns:
            np.ndarray: _description_
        """
        w, h = ksize
        half_w, half_h = w * 0.5, h * 0.5
        half_max = max(half_w, half_h)
        gaussian2d_heatmap = np.zeros((h, w), np.uint8)
        for y in range(h):
            for x in range(w):
                distance_from_center = np.linalg.norm(np.array([y - half_h, x - half_w]))
                distance_from_center = dratio * distance_from_center / half_max
                scaled_gaussian_prob = math.exp(-0.5 * (distance_from_center ** 2))
                gaussian2d_heatmap[y, x] = np.clip(scaled_gaussian_prob * 255, 0, 255)
        
        return gaussian2d_heatmap
    
    def perspective_transform(
        self,
        image:np.ndarray,
        bbox:tuple[int, int, int, int],
    ) -> np.ndarray:
        """射影変換

        Args:
            image (np.ndarray): _description_
            bbox (tuple[int]): (xmin, ymin, xmax, ymax)

        Returns:
            np.ndarray: _description_
        """
        xmin, ymin, xmax, ymax = bbox
        
        h1, w1 = image.shape
        h2, w2 = (ymax - ymin, xmax - xmin)
        
        pts1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32)
        pts2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32)
        
        # 画像をbboxサイズに射影変換
        M = cv2.getPerspectiveTransform(pts1, pts2)
        image = cv2.warpPerspective(image, M, (w2, h2), flags=cv2.INTER_LINEAR)
        
        return image
    
    def __call__(
        self,
        image_size:tuple[int, int],
        bboxes:list[tuple[int, int, int, int]],
    ) -> np.ndarray:
        """_summary_

        Args:
            image_size (tuple[int]): _description_
            bboxes (list[tuple[int]]): _description_

        Returns:
            np.ndarray: _description_
        """
        # blend:addな挙動をするので16bitにしている
        image = np.zeros(image_size, dtype=np.uint16)
        g2dheatmap = self.gaussian2d.copy()
        
        for bbox in bboxes:
            # ヒートマップをbboxサイズに変換
            warped = self.perspective_transform(g2dheatmap, bbox)
            
            xmin, ymin, xmax, ymax = bbox
            image[ymin:ymax, xmin:xmax] += warped
        
        return np.clip(image, 0, 255).astype(np.uint8)
