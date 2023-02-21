import os
import copy
from pathlib import Path
import torch


def my_app(
    checkpoint_path:str,
    output_dir:str,
    model_name:str,
):
    """チェックポイントから学習済みモデルを作成

    Args:
        checkpoint_path (str): チェックポイントパス
        output_dir (str): 出力先のディレクトリ
        model_name (str): モデル名
    """
    device = torch.device("cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 不必要なパラメータの削除
    for key in copy.copy(list(checkpoint)):
        if key == "state_dict":
            for val in list(checkpoint[key]):
                checkpoint[key][val.replace("model.", "")] = checkpoint[key].pop(val)

        elif key == "hyper_parameters":
            for val in list(checkpoint[key]):
                if val not in ["pretrained", "freeze"]:
                    del checkpoint[key][val]
        
        else:
            del checkpoint[key]
    
    # ナンバリング
    output_dir:Path = Path(output_dir)
    output_version = [int(dir.stem.split("_v")[1]) for dir in output_dir.glob("*.pth") if dir.name.startswith(model_name)]
    output_version = max(output_version) + 1 if len(output_version) > 0 else 0
    
    # 出力先の作成
    output_path = Path(output_dir) / f"{model_name}_v{output_version}.pth"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存
    torch.save(checkpoint, str(output_path))
    
    # ファイルサイズ
    before_size = os.path.getsize(checkpoint_path)
    after_size = os.path.getsize(str(output_path))
    
    # デバッグ表示
    print(str(output_path))
    print(f"File size (before > after): {before_size/1024/1024:.3f} > {after_size/1024/1024:.3f} MB")


if __name__ == "__main__":
    my_app(
        r"E:\ReinVisionOCR\post-craft\log_logs\version_0\checkpoints\last.ckpt",
        r"E:\ReinVisionOCR\resources\pretrained\msgothic",
        "post_craft",
    )
