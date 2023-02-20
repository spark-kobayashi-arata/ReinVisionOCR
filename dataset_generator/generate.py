import yaml
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import shutil

from common import *
from character import *


def my_app(config_path:str):
    """PostCRAFTのデータセットの生成

    Args:
        config_path (str): _description_
    """
    with open(config_path, mode="r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        config = CharacterConfig(**config)
    
    with ProcessPoolExecutor(config.max_workers) as executor:
        for stage in StageType:
            # 保存先
            stage_dir = config.output_dir / str(stage)
            stage_dir.mkdir(parents=True, exist_ok=True)
            
            futures = [
                executor.submit(
                    create_character_image,
                    *character_params,
                    stage_dir,
                    idx,
                )
                for idx, character_params in enumerate(config.create_iter(stage))
            ]
            
            for future in tqdm(futures, desc=f"{stage}", postfix=f"v_num={config.output_version}"):
                future.result()
    
    # 設定ファイルのコピー
    shutil.copyfile(config_path, str(config.output_dir / "config.yaml"))


if __name__ == "__main__":
    my_app(r"E:\ReinVisionOCR\post-craft\dataset_generator\config.yaml")
