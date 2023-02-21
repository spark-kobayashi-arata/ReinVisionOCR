from pathlib import Path
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def my_app(remove_dir:str):
    """データセットの高速削除

    Args:
        remove_dir (str): ディレクトリを指定
    """
    if not os.path.exists(remove_dir):
        return
    
    with ThreadPoolExecutor(os.cpu_count()) as executor:
        futures = [
            executor.submit(
                os.remove,
                str(path),
            )
            for path in Path(remove_dir).glob("**/*.pickle")
        ]
        
        futures += [
            executor.submit(
                os.remove,
                str(path),
            )
            for path in Path(remove_dir).glob("**/*.png")
        ]
        
        for future in tqdm(futures, desc=remove_dir):
            future.result()
    
    # ファイルの削除が完了したらディレクトリごと削除
    shutil.rmtree(remove_dir)


if __name__ == "__main__":
    for i in range(9, 10):
        my_app(fr"E:\ReinVisionOCR\post-craft\dataset\version_{i}")
