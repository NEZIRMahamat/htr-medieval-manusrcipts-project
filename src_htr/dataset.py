"""
dataset.py for the HTR project on medieval manuscripts. (TrOCR)
"""
from datasets import load_from_disk

def load_data_from_dir(disk_path) -> dict:
    """
    Load a HuggingFace DatasetDict saved with `save_to_disk`.
    """
    return load_from_disk(disk_path)

