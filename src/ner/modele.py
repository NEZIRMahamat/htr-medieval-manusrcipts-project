"""
Fine-tuning de CamemBERT avec LoRA pour la NER sur manuscrits médiévaux.

Architecture :
  - Modèle de base : camembert-base (RoBERTa français, 110M paramètres)
  - Adaptation LoRA : appliquée aux couches d'attention
  - Tête de classification : couche linéaire → 11 labels BIO

Références :
  - CamemBERT : Martin et al. (2019). CamemBERT: a Tasty French Language Model.
  - LoRA : Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
"""

from __future__ import annotations

from pathlib import Path

import torch
import numpy as np
from transformers import (
    CamembertTokenizerFast,
    CamembertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from peft import get_peft_model, LoraConfig, TaskType

import random
import json
from datetime import datetime, timezone
from pathlib import Path

from src.ner.dataset import (
    DatasetNERMedieval,
    EXEMPLES_ANNOTATION,
    LABEL2ID,
    ID2LABEL,
    LABELS,
)


def fixer_seeds(seed: int = 42) -> None:
    """Fixe toutes les sources d'aléatoire."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_experiment(entry: dict) -> None:
    """Enregistre une expérience dans le journal JSONL."""
    Path("experiments").mkdir(exist_ok=True)
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open("experiments/journal.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


MODELE_BASE = "camembert-base"


# ── Chargement du modèle ───────────────────────────────────────────────────────


def charger_camembert_lora(
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
):
    """Charge CamemBERT et applique LoRA pour la NER.

    Args:
        lora_r: Rang LoRA. 8 = bon compromis vitesse/performance.
        lora_alpha: Facteur d'échelle (généralement 4×r).
        lora_dropout: Dropout sur les couches LoRA.

    Returns:
        Tuple (modèle avec LoRA, tokeniseur).

    Example:
        >>> modele, tokeniseur = charger_camembert_lora(lora_r=8)
        >>> modele.print_trainable_parameters()
        # trainable params: 147,456 || all params: 110,780,427
    """
    tokeniseur = CamembertTokenizerFast.from_pretrained(MODELE_BASE)

    modele = CamembertForTokenClassification.from_pretrained(
        MODELE_BASE,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Configuration LoRA sur les couches d'attention
    config_lora = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["query", "value"],
        bias="none",
    )

    modele = get_peft_model(modele, config_lora)
    modele.print_trainable_parameters()

    return modele, tokeniseur


# ── Métriques ─────────────────────────────────────────────────────────────────


def calculer_metriques(predictions_et_labels: tuple) -> dict:
    """Calcule le F1 par type d'entité pour l'évaluation NER.

    Args:
        predictions_et_labels: Tuple (logits, labels) produit par le Trainer.

    Returns:
        Dictionnaire avec F1 global et par type d'entité.
    """
    logits, labels = predictions_et_labels
    predictions = np.argmax(logits, axis=-1)

    # Ignorer les tokens spéciaux (-100)
    vrais_labels = [
        [ID2LABEL[l] for l in label_seq if l != -100]
        for label_seq in labels
    ]
    vraies_preds = [
        [ID2LABEL[p] for p, l in zip(pred_seq, label_seq) if l != -100]
        for pred_seq, label_seq in zip(predictions, labels)
    ]

    # Calcul F1 par entité
    try:
        from seqeval.metrics import f1_score, precision_score, recall_score
        f1 = f1_score(vrais_labels, vraies_preds)
        precision = precision_score(vrais_labels, vraies_preds)
        recall = recall_score(vrais_labels, vraies_preds)
    except ImportError:
        # Fallback sans seqeval
        f1 = precision = recall = 0.0

    return {
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


# ── Entraînement ──────────────────────────────────────────────────────────────


def entrainer_ner(
    exemples_train: list | None = None,
    exemples_val: list | None = None,
    dossier_sortie: str | Path = "checkpoints/ner_camembert",
    lora_r: int = 8,
    learning_rate: float = 2e-4,
    nb_epochs: int = 10,
    batch_size: int = 8,
    seed: int = 42,
) -> tuple:
    """Fine-tune CamemBERT+LoRA pour la NER médiévale.

    Args:
        exemples_train: Exemples annotés pour l'entraînement.
                        Si None, utilise EXEMPLES_ANNOTATION (données intégrées).
        exemples_val: Exemples pour la validation.
                      Si None, utilise les 2 derniers exemples de train.
        dossier_sortie: Dossier pour les checkpoints.
        lora_r: Rang LoRA.
        learning_rate: Taux d'apprentissage (2e-4 recommandé pour LoRA NER).
        nb_epochs: Nombre d'epochs.
        batch_size: Taille de batch.
        seed: Graine aléatoire.

    Returns:
        Tuple (modèle fine-tuné, tokeniseur).

    Example:
        >>> modele, tok = entrainer_ner(dossier_sortie="checkpoints/ner_r8")
    """
    fixer_seeds(seed)

    # Données par défaut
    if exemples_train is None:
        exemples_train = EXEMPLES_ANNOTATION[:-2]
    if exemples_val is None:
        exemples_val = EXEMPLES_ANNOTATION[-2:]

    modele, tokeniseur = charger_camembert_lora(lora_r=lora_r)

    dataset_train = DatasetNERMedieval(exemples_train, tokeniseur)
    dataset_val = DatasetNERMedieval(exemples_val, tokeniseur)

    collateur = DataCollatorForTokenClassification(tokeniseur)

    args = TrainingArguments(
        output_dir=str(dossier_sortie),
        num_train_epochs=nb_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=f"{dossier_sortie}/logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=seed,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=modele,
        args=args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=collateur,
        compute_metrics=calculer_metriques,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # Journal
    log_experiment({
        "etape": "fine_tuning_ner_camembert",
        "hyperparams": {
            "lora_r": lora_r,
            "learning_rate": learning_rate,
            "nb_epochs": nb_epochs,
            "batch_size": batch_size,
            "modele_base": MODELE_BASE,
        },
        "metriques": {
            "f1_finale": trainer.state.best_metric,
            "nb_exemples_train": len(exemples_train),
        },
    })

    return modele, tokeniseur


# ── Inférence ──────────────────────────────────────────────────────────────────


def extraire_entites(
    texte: str,
    modele,
    tokeniseur,
    seuil_confiance: float = 0.7,
) -> list[dict]:
    """Extrait les entités nommées d'un texte normalisé.

    Args:
        texte: Texte normalisé (sortie du module de normalisation).
        modele: CamemBERT fine-tuné.
        tokeniseur: Tokeniseur CamemBERT.
        seuil_confiance: Score minimum pour retenir une entité.

    Returns:
        Liste de dicts {"texte", "type", "debut", "fin", "confiance"}.

    Example:
        >>> entites = extraire_entites("le roy Phelippe de France", modele, tok)
        >>> entites
        [{"texte": "Phelippe", "type": "PER", "debut": 7, "fin": 15, "confiance": 0.94},
         {"texte": "France", "type": "LOC", "debut": 19, "fin": 25, "confiance": 0.91}]
    """
    modele.eval()
    tokens = texte.split()

    encodage = tokeniseur(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        sorties = modele(**encodage)

    logits = sorties.logits[0]
    probs = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(probs, dim=-1)
    confiances = probs.max(dim=-1).values

    # Aligner les prédictions sur les tokens originaux
    word_ids = encodage.word_ids(batch_index=0)
    labels_par_mot = {}

    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id not in labels_par_mot:
            labels_par_mot[word_id] = {
                "label": ID2LABEL[predictions[idx].item()],
                "confiance": confiances[idx].item(),
            }

    # Reconstruire les entités à partir du schéma BIO
    entites = []
    entite_courante = None

    for i, token in enumerate(tokens):
        if i not in labels_par_mot:
            continue

        label = labels_par_mot[i]["label"]
        conf = labels_par_mot[i]["confiance"]

        if label.startswith("B-"):
            # Sauvegarder l'entité précédente
            if entite_courante and entite_courante["confiance"] >= seuil_confiance:
                entites.append(entite_courante)

            type_entite = label[2:]  # "B-PER" → "PER"
            entite_courante = {
                "texte": token,
                "type": type_entite,
                "tokens": [token],
                "confiance": conf,
            }

        elif label.startswith("I-") and entite_courante:
            entite_courante["texte"] += " " + token
            entite_courante["tokens"].append(token)
            entite_courante["confiance"] = min(entite_courante["confiance"], conf)

        else:
            # O : fin de l'entité courante
            if entite_courante and entite_courante["confiance"] >= seuil_confiance:
                entites.append(entite_courante)
            entite_courante = None

    # Dernière entité
    if entite_courante and entite_courante["confiance"] >= seuil_confiance:
        entites.append(entite_courante)

    # Nettoyer (enlever "tokens" de la sortie)
    for e in entites:
        e.pop("tokens", None)
        e["confiance"] = round(e["confiance"], 4)

    return entites


def annoter_page(
    lignes_normalisees: list[dict],
    modele,
    tokeniseur,
) -> list[dict]:
    """Annote toutes les lignes d'une page avec les entités détectées.

    Args:
        lignes_normalisees: Sortie du module de normalisation.
        modele: CamemBERT fine-tuné.
        tokeniseur: Tokeniseur CamemBERT.

    Returns:
        Liste de dicts avec les entités ajoutées.

    Example:
        >>> resultats = annoter_page(lignes, modele, tokeniseur)
        >>> resultats[0]["entites"]
        [{"texte": "Phelippe", "type": "PER", "confiance": 0.94}]
    """
    resultats = []

    for ligne in lignes_normalisees:
        texte = ligne.get("texte_normalise", ligne.get("transcription", ""))
        entites = extraire_entites(texte, modele, tokeniseur)

        resultats.append({
            **ligne,
            "entites": entites,
            "nb_entites": len(entites),
        })

    return resultats