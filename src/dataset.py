from dotenv import load_dotenv
from huggingface_hub import login 
from datasets import load_dataset, load_from_disk, Image
from matplotlib import pyplot as plt
import tensorflow as tf
import os

# Load env
load_dotenv()

data_dir = "data"
outputs_dir = "outputs"

## ------ Téléchargement du dataset CATMuS depuis Hugging Face Hub et sauvegarde sur local disk

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

# Save the dataset to local directory
def save_dataset_to_dir(dataset, save_path) -> None:
    dataset.save_to_disk(save_path)

def load_save_catmus_dataset(medieval_data_path, seg_data_path):

  os.makedirs(medieval_data_path, exist_ok=True)
  os.makedirs(seg_data_path, exist_ok=True)
  
  # Medieval data
  dataset_medieval = load_medieval_dataset()
  save_dataset_to_dir(dataset_medieval, medieval_data_path)

  # Segmentation data
  seg_dataset = load_medieval_segmentation_dataset()
  save_dataset_to_dir(seg_dataset, seg_data_path)


def main_load_dataset_from_hf():
    # Load & save the dataset    
    save_path_medieval = os.path.join(data_dir, "medieval_data") 
    print(f"Medieval path : {save_path_medieval}")

    save_path_segm = os.path.join(data_dir, "segment_data")
    print(f"Segmentation path : {save_path_segm}")

    load_save_catmus_dataset(save_path_medieval, save_path_segm)



## ------ Chargement du dataset CATMuS HF depuis le disque local

def load_data_from_dir(disk_path) -> dict:
    """
    Load a HuggingFace DatasetDict saved with `save_to_disk`.
    """
    return load_from_disk(disk_path)


## ------ AUDIT DATASET
def audit_dataset(dataset):
    """
    Inspect the CATMuS segmentation DatasetDict structure.
    """
    raw_dataset = dataset.cast_column("image", Image(decode=False))

    print("Nombre d'exemples par split :")
    for split in dataset.keys():
        print(f"{split}: {len(dataset[split])} exemples")

    print("\nColonnes disponibles :")
    for column in dataset["train"].column_names:
        print(column)

    print("\nInspection de quelques pages :")
    for i in range(5):
        example = dataset["train"][i]
        raw_example = raw_dataset["train"][i]
        image = example["image"]
        image_path = raw_example["image"].get("path") or ""
        image_name = image_path.split("/")[-1].split("\\")[-1]
        mask_name = f"{image_name.rsplit('.', 1)[0]}_mask.png"
        objects = example["objects"]
        object_types = objects.get("type", [])
        categories = objects.get("category", [])

        print(f"Exemple {i + 1}:")
        print(f"  Taille image: {image.size}")
        print(f"  Nom du fichier image: {image_name}")
        print(f"  Nom mask temporaire: {mask_name}")
        print(f"  Nombre d'objets: {len(object_types)}")
        print(f"  Types: {sorted(set(object_types))}")
        print(f"  Categories: {sorted(set(categories))}")

        # Intéressant pour manipuler les polygones pour la segmentation (ou juste checks info ++)
        # id = objects.get("id", [])
        # bboxes = objects.get("bbox", [])
        # polygons = objects.get("polygons", [])
        # parent = objects.get("parent", [])
        # Sortie
        # Exemple : 4
        #   Types: ['block', 'line']
        #   Categories: ['DefaultLine', 'DropCapitalZone', 'HeadingLine', 'MainZone', 'NumberingZone', 'RunningTitleZone'] (et autres aussi, car test audit sur 5 images seulement)
        #  Polygones : [[x1, y1, x2, y2, x3, y3, ...], [...], ...] (pour chaque objet)
        # Exemple 5:
        #   Taille image: (3018, 3836)
        #   Nom du fichier image: page-005-of-075.jpg
        #   Nom mask temporaire: train_000004_page-005-of-075_mask.png
        #   Nombre d'objets: 97
        #   Types: ['block', 'line']
        #   Categories: ['DefaultLine', 'MainZone', 'MarginTextZone', 'RunningTitleZone']

def visualize_dataset_audit(dataset, num_examples=5):
    """
    Visualize examples with CATMuS line polygons overlaid on images.
    """    

    for i in range(num_examples):
        example = dataset["train"][i]
        image = example["image"]
        objects = example["objects"]

        plt.figure(figsize=(10, 5))
        plt.imshow(image)

        for polygon, obj_type in zip(objects.get("polygons", []), objects.get("type", [])):
            if obj_type != "line" or not polygon:
                continue

            points = list(zip(polygon[0::2], polygon[1::2]))
            xs = [point[0] for point in points] + [points[0][0]]
            ys = [point[1] for point in points] + [points[0][1]]
            plt.plot(xs, ys, color="red")

        plt.title(f"Exemple {i + 1}")
        plt.axis("off")
        plt.show()

def main_audit():
    dataset_path = os.path.join(data_dir, "segment_data")
    dataset = load_data_from_dir(dataset_path)
    audit_dataset(dataset)
    visualize_dataset_audit(dataset, num_examples=5)

## ------ SCELLAGE DATASET TEST

# étape final : sceller le dataset test pour éviter les fuites de données
# Evaluation du modèle sur le dataset test (only après l'entrainement et la validation)


if __name__ == "__main__":
    main_audit()
    print("Audit completed successfully.")