import yaml
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import shutil
import json

from common import *
from binary_character import *


def my_app(config_path:str):
    """CoAtNetのデータセット生成

    Args:
        config_path (str): _description_
    """
    with open(config_path, mode="r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        config = BinaryCharacterConfig(**config)
    
    with ProcessPoolExecutor(config.max_workers) as executor:
        for stage in StageType:
            # 保存先
            stage_dir = config.output_dir / str(stage)
            stage_dir.mkdir(parents=True, exist_ok=True)
            
            futures = [
                executor.submit(
                    create_binary_character_image,
                    *params,
                    stage_dir,
                    idx,
                )
                for idx, params in enumerate(config.create_iter(stage))
            ]
            
            stage_debug_dir = config.output_dir / f"{stage}_debug"
            stage_debug_dir.mkdir(parents=True, exist_ok=True)
            
            with TileImageGenerator(64, 64, stage_debug_dir, "image") as tig:
                for future in tqdm(futures, desc=f"{stage}", postfix=f"v_num={config.output_version}"):
                    tig.add(future.result())
    
    with open(str(config.output_dir / "codes.json"), mode="w", encoding="utf-8") as f:
        json.dump({label:char for label, char in enumerate(config.code_list)}, f, indent=2, ensure_ascii=False, sort_keys=False)
    
    # 設定ファイルのコピー
    shutil.copyfile(config_path, str(config.output_dir / "config.yaml"))


if __name__ == "__main__":
    my_app(r"E:\ReinVisionOCR\coatnet\dataset_generator\config.yaml")
