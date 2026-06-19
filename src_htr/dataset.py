from __future__ import annotations
import numpy as np
from PIL import Image
from datasets import load_from_disk
from torch.utils.data import Dataset

def load_data_from_dir(disk_path):
    return load_from_disk(disk_path)

def extraire_exemples_htr(dataset, split=None, max_exemples=None):
    split = split or chr(116)+chr(114)+chr(97)+chr(105)+chr(110)
    data = dataset[split]
    if max_exemples:
        data = data.select(range(min(max_exemples, len(data))))
    exemples = []
    for item in data:
        image = item.get(chr(105)+chr(109)+chr(97)+chr(103)+chr(101))
        transcription = item.get(chr(116)+chr(101)+chr(120)+chr(116), item.get(chr(116)+chr(114)+chr(97)+chr(110)+chr(115)+chr(99)+chr(114)+chr(105)+chr(112)+chr(116)+chr(105)+chr(111)+chr(110), chr(32)))
        if image is None or not transcription:
            continue
        if not isinstance(image, Image.Image):
            try:
                image = Image.fromarray(__import__(chr(110)+chr(117)+chr(109)+chr(112)+chr(121)).array(image)).convert(chr(82)+chr(71)+chr(66))
            except Exception:
                continue
        exemples.append({chr(105)+chr(109)+chr(97)+chr(103)+chr(101): image.convert(chr(82)+chr(71)+chr(66)), chr(116)+chr(114)+chr(97)+chr(110)+chr(115)+chr(99)+chr(114)+chr(105)+chr(112)+chr(116)+chr(105)+chr(111)+chr(110): transcription})
    return exemples

class DatasetHTRMedieval(Dataset):
    def __init__(self, exemples, processeur, max_longueur=128):
        self.exemples = exemples
        self.processeur = processeur
        self.max_longueur = max_longueur
    def __len__(self):
        return len(self.exemples)
    def __getitem__(self, idx):
        exemple = self.exemples[idx]
        image = exemple[chr(105)+chr(109)+chr(97)+chr(103)+chr(101)]
        transcription = exemple[chr(116)+chr(114)+chr(97)+chr(110)+chr(115)+chr(99)+chr(114)+chr(105)+chr(112)+chr(116)+chr(105)+chr(111)+chr(110)]
        pixel_values = self.processeur(images=image, return_tensors=chr(112)+chr(116)).pixel_values.squeeze(0)
        labels = self.processeur.tokenizer(text=transcription, max_length=self.max_longueur, padding=chr(109)+chr(97)+chr(120)+chr(95)+chr(108)+chr(101)+chr(110)+chr(103)+chr(116)+chr(104), truncation=True, return_tensors=chr(112)+chr(116)).input_ids.squeeze(0)
        labels[labels == self.processeur.tokenizer.pad_token_id] = -100
        return {chr(112)+chr(105)+chr(120)+chr(101)+chr(108)+chr(95)+chr(118)+chr(97)+chr(108)+chr(117)+chr(101)+chr(115): pixel_values, chr(108)+chr(97)+chr(98)+chr(101)+chr(108)+chr(115): labels}