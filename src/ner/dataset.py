"""
Dataset PyTorch pour le fine-tuning NER avec CamemBERT.

Le schéma d'annotation BIO utilisé :
  - O       : pas une entité
  - B-PER   : début d'une personne
  - I-PER   : suite d'une personne
  - B-LOC   : début d'un lieu
  - I-LOC   : suite d'un lieu
  - B-DATE  : début d'une date
  - I-DATE  : suite d'une date
  - B-ORG   : début d'une organisation
  - I-ORG   : suite d'une organisation
  - B-TITLE : début d'un titre (duc, comte, roi...)
  - I-TITLE : suite d'un titre
"""

from __future__ import annotations

import json
from pathlib import Path
from torch.utils.data import Dataset
import torch


# ── Schéma BIO ─────────────────────────────────────────────────────────────────

LABELS = [
    "O",
    "B-PER", "I-PER",
    "B-LOC", "I-LOC",
    "B-DATE", "I-DATE",
    "B-ORG", "I-ORG",
    "B-TITLE", "I-TITLE",
]

LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


# ── Données d'entraînement annotées manuellement ──────────────────────────────
# Format : liste de (tokens, labels)
# Ces exemples servent à fine-tuner CamemBERT sur le vieux français médiéval

EXEMPLES_ANNOTATION = [
    (
        ["le", "roy", "Phelippe", "de", "France", "sont", "parti", "de", "Paris"],
        ["O", "O", "B-PER", "O", "B-LOC", "O", "O", "O", "B-LOC"]
    ),
    (
        ["monseigneur", "Jehan", "duc", "de", "Berry", "et", "ses", "barons"],
        ["O", "B-PER", "I-PER", "I-PER", "I-PER", "O", "O", "O"]
    ),
    (
        ["monseigneur", "Loys", "duc", "d", "Anjou", "son", "frere"],
        ["O", "B-PER", "I-PER", "I-PER", "I-PER", "O", "O"]
    ),
    (
        ["vers", "la", "cite", "de", "Lyon", "le", "xviij", "jour", "de", "marz"],
        ["O", "O", "O", "O", "B-LOC", "O", "B-DATE", "I-DATE", "I-DATE", "I-DATE"]
    ),
    (
        ["le", "conte", "de", "Flandres", "et", "ses", "barons"],
        ["O", "B-TITLE", "I-TITLE", "I-TITLE", "O", "O", "O"]
    ),
    (
        ["en", "la", "cite", "de", "Bruges", "por", "parler", "au", "conte"],
        ["O", "O", "O", "O", "B-LOC", "O", "O", "O", "O"]
    ),
    (
        ["le", "royaume", "de", "France", "et", "la", "paix", "de", "Flandres"],
        ["O", "O", "O", "B-LOC", "O", "O", "O", "O", "B-LOC"]
    ),
    (
        ["en", "lan", "de", "grace", "mil", "trois", "cenz", "et", "dis"],
        ["O", "B-DATE", "I-DATE", "I-DATE", "I-DATE", "I-DATE", "I-DATE", "I-DATE", "I-DATE"]
    ),
    (
        ["Jhesu", "Crist", "et", "la", "vierge", "Marie"],
        ["B-PER", "I-PER", "O", "O", "B-PER", "I-PER"]
    ),
    (
        ["par", "le", "conseil", "du", "roy", "et", "de", "ses", "nobles", "hommes"],
        ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]
    ),
    (
        ["messages", "en", "la", "cite", "de", "Bruges"],
        ["O", "O", "O", "O", "O", "B-LOC"]
    ),
    (
        ["le", "roy", "de", "France", "son", "seigneur"],
        ["O", "O", "O", "B-LOC", "O", "O"]
    ),
]


# ── Dataset PyTorch ────────────────────────────────────────────────────────────


class DatasetNERMedieval(Dataset):
    """Dataset pour le fine-tuning NER sur des transcriptions médiévales.

    Gère l'alignement entre les tokens originaux et les sous-tokens
    produits par le tokeniseur CamemBERT (WordPiece).

    Args:
        exemples: Liste de tuples (tokens, labels).
        tokeniseur: Tokeniseur CamemBERT.
        longueur_max: Longueur maximale de séquence.

    Example:
        >>> dataset = DatasetNERMedieval(EXEMPLES_ANNOTATION, tokeniseur)
        >>> len(dataset)
        12
    """

    def __init__(
        self,
        exemples: list[tuple[list[str], list[str]]],
        tokeniseur,
        longueur_max: int = 128,
    ) -> None:
        self.exemples = exemples
        self.tokeniseur = tokeniseur
        self.longueur_max = longueur_max

    def __len__(self) -> int:
        return len(self.exemples)

    def __getitem__(self, idx: int) -> dict:
        tokens, labels = self.exemples[idx]

        # Tokenisation avec alignement des labels
        encodage = self.tokeniseur(
            tokens,
            is_split_into_words=True,
            max_length=self.longueur_max,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Aligner les labels sur les sous-tokens
        # Les sous-tokens d'un même mot reçoivent -100 (ignorés dans la loss)
        word_ids = encodage.word_ids(batch_index=0)
        labels_alignes = []
        word_id_precedent = None

        for word_id in word_ids:
            if word_id is None:
                # Token spécial [CLS] ou [SEP]
                labels_alignes.append(-100)
            elif word_id != word_id_precedent:
                # Premier sous-token du mot → label réel
                labels_alignes.append(LABEL2ID[labels[word_id]])
            else:
                # Sous-token suivant → ignoré
                labels_alignes.append(-100)
            word_id_precedent = word_id

        return {
            "input_ids": encodage["input_ids"].squeeze(0),
            "attention_mask": encodage["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels_alignes, dtype=torch.long),
        }