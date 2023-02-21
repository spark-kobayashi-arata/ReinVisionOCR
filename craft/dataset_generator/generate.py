from dataclasses import dataclass
import yaml
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import shutil

from common import *
from word import *
from text import *
from game import *


@dataclass
class Config(CommonConfig):
    """Word, Text, Gameの設定を内包
    """
    word:WordConfig
    text:TextConfig
    game:GameConfig
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.word = WordConfig(self.common, **kwargs["word"])
        self.text = TextConfig(self.common, **kwargs["text"])
        self.game = GameConfig(self.common, **kwargs["game"])
    
    @property
    def common(self) -> CommonConfig:
        """共通設定を取得

        Returns:
            CommonConfig: _description_
        """
        return super()


def my_app(config_path:str):
    """CRAFTのデータセットの生成

    Args:
        config_path (str): _description_
    """
    with open(config_path, mode="r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        config = Config(**config)
    
    with ProcessPoolExecutor(config.max_workers) as executor:
        for stage in StageType:
            # 保存先
            stage_dir = config.output_dir / str(stage)
            stage_dir.mkdir(parents=True, exist_ok=True)
            
            futures = [
                executor.submit(
                    create_word_image,
                    *word_params,
                    *config.font_style,
                    stage_dir,
                    idx,
                )
                for idx, word_params in enumerate(config.word.create_iter(stage))
            ]
            
            futures += [
                executor.submit(
                    create_word_image,
                    *text_params,
                    *config.font_style,
                    stage_dir,
                    idx,
                )
                for idx, text_params in enumerate(config.text.create_iter(stage), len(futures))
            ]
            
            futures += [
                executor.submit(
                    create_game_image,
                    *game_params,
                    *config.font_style,
                    stage_dir,
                    idx,
                )
                for idx, game_params in enumerate(config.game.create_iter(stage), len(futures))
            ]
            
            for future in tqdm(futures, desc=f"{stage}", postfix=f"v_num={config.output_version}"):
                future.result()
    
    # 設定ファイルのコピー
    shutil.copyfile(config_path, str(config.output_dir / "config.yaml"))


if __name__ == "__main__":
    my_app(r"E:\ReinVisionOCR\craft\dataset_generator\config.yaml")
