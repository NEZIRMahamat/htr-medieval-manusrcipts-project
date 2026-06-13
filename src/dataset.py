from dotenv import load_dotenv
from huggingface_hub import login 
from datasets import load_dataset
import os

# Load env
load_dotenv()

data_dir = "data"


# Login to Hugging Face Hub
def login_to_hf() -> None:
    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)

# Loader medieval dataset
def load_medieval_dataset() -> dict:
    """
    Load the medieval dataset from Hugging Face Hub.
    Pour le HTR : dataset "medieval".
    """
    login_to_hf()
    dataset = load_dataset("CATMuS/medieval", verification_mode="no_checks")
    return dataset

# Loader segmentation dataset
def load_medieval_segmentation_dataset() -> dict:
    """
    Load the medieval segmentation dataset from Hugging Face Hub.
    Pour la segmentation : dataset "medieval-segmentation".
    """
    login_to_hf()
    dataset = load_dataset("CATMuS/medieval-segmentation", verification_mode="no_checks")
    return dataset

# Save the dataset to a local directory
def save_dataset(dataset, save_path) -> None:
    dataset.save_to_disk(save_path)

def load_save_catmus_dataset(medieval_data_path, seg_data_path):

  os.makedirs(medieval_data_path, exist_ok=True)
  os.makedirs(seg_data_path, exist_ok=True)
  # Medieval data
  dataset_medieval = load_medieval_dataset()
  save_dataset(dataset_medieval, medieval_data_path)

  # Segmentation data
  seg_dataset = load_medieval_segmentation_dataset()
  save_dataset(seg_dataset, seg_data_path)


def main_dataset():
    # Load & save the dataset
    
    save_path_medieval = os.path.join(data_dir, "medieval_data")  # Pour le HTR
    print(f"Medieval path : {save_path_medieval}")

    save_path_segm = os.path.join(data_dir, "segment_data") # Pour la segmentation
    print(f"Segmentation path : {save_path_segm}")

    load_save_catmus_dataset(save_path_medieval, save_path_segm)


if __name__ == "__main__":
    main_dataset()