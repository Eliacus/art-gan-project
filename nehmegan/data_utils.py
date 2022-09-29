import os
from pathlib import Path
from typing import Tuple

from nehmegan import ROOT


def verify_and_get_dataset_paths(dataset_name: str) -> Tuple[Path, Path]:
    dataset_root_path = Path.joinpath(ROOT, "data", dataset_name)
    dataset_images_path = Path.joinpath(dataset_root_path, "images")
    _verify_dataset_exists(dataset_name, dataset_root_path, dataset_images_path)
    return dataset_root_path, dataset_images_path


def _verify_dataset_exists(
    dataset_name,
    dataset_root_path,
    dataset_images_path,
) -> None:
    """Verifies that the provided dataset string points to an existing dataset"""
    if not os.path.exists(dataset_root_path):
        raise FileNotFoundError(
            f"Could not find dataset called '{dataset_name}' under data/"
        )
    elif not os.path.exists(dataset_images_path):
        raise FileNotFoundError(
            f"Images for dataset '{dataset_name}' are not located in a subdirectory images/"
        )
