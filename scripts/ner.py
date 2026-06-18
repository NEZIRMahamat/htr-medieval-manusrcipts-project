"""
Script NER : fine-tuning CamemBERT+LoRA et extraction d'entités.

Usage :
    # Entraîner le modèle
    python scripts/ner.py --mode train --output checkpoints/ner/

    # Extraire les entités sur les transcriptions normalisées
    python scripts/ner.py --mode predict --input data/normalise.json --output data/entites.json --checkpoint checkpoints/ner/

    # Les deux d'un coup
    python scripts/ner.py --mode all --input data/normalise.json --output data/entites.json
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

from src.ner.modele import (
    charger_camembert_lora,
    entrainer_ner,
    annoter_page,
    MODELE_BASE,
)


def mode_train(args):
    """Lance le fine-tuning CamemBERT+LoRA."""
    print("🚀 Fine-tuning CamemBERT+LoRA pour la NER médiévale...")
    print(f"   Modèle de base : {MODELE_BASE}")
    print(f"   LoRA r={args.lora_r}")
    print(f"   Epochs : {args.epochs}")

    modele, tokeniseur = entrainer_ner(
        dossier_sortie=args.checkpoint,
        lora_r=args.lora_r,
        learning_rate=args.lr,
        nb_epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Sauvegarder
    chemin = Path(args.checkpoint)
    chemin.mkdir(parents=True, exist_ok=True)
    modele.save_pretrained(str(chemin))
    tokeniseur.save_pretrained(str(chemin))
    print(f"✅ Modèle sauvegardé : {chemin}")

    return modele, tokeniseur


def mode_predict(args, modele=None, tokeniseur=None):
    """Extrait les entités sur les transcriptions normalisées."""
    from transformers import CamembertTokenizerFast, CamembertForTokenClassification
    from peft import PeftModel

    # Charger le modèle si pas déjà en mémoire
    if modele is None:
        print(f"📂 Chargement du modèle : {args.checkpoint}")
        chemin = Path(args.checkpoint)
        if chemin.exists():
            base = CamembertForTokenClassification.from_pretrained(str(chemin))
            modele = PeftModel.from_pretrained(base, str(chemin))
            tokeniseur = CamembertTokenizerFast.from_pretrained(str(chemin))
        else:
            print(f"⚠️  Checkpoint non trouvé. Utilisation du modèle de base (sans fine-tuning).")
            modele, tokeniseur = charger_camembert_lora(lora_r=args.lora_r)

    # Charger les transcriptions normalisées
    print(f"📂 Chargement : {args.input}")
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    lignes = data.get("lignes", [])
    print(f"  ✅ {len(lignes)} lignes chargées")

    # Annoter
    print("🔍 Extraction des entités...")
    resultats = annoter_page(lignes, modele, tokeniseur)

    # Statistiques
    toutes_entites = [e for r in resultats for e in r.get("entites", [])]
    stats_types: dict[str, int] = {}
    for e in toutes_entites:
        t = e["type"]
        stats_types[t] = stats_types.get(t, 0) + 1

    print(f"\n📊 Entités extraites : {len(toutes_entites)}")
    for type_ent, count in sorted(stats_types.items()):
        print(f"   {type_ent:10} : {count}")

    # Export
    sortie = {
        "metadata": {
            "nb_lignes": len(resultats),
            "nb_entites_total": len(toutes_entites),
            "entites_par_type": stats_types,
        },
        "lignes": resultats,
    }

    chemin_sortie = Path(args.output)
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
    chemin_sortie.write_text(
        json.dumps(sortie, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n✅ Résultats exportés → {chemin_sortie}")

    # Exemples qualitatifs
    print("\n📖 Exemples de détections :")
    for r in resultats[:5]:
        if r.get("entites"):
            print(f"   Texte : {r.get('texte_normalise', r.get('transcription', ''))[:60]}")
            for e in r["entites"]:
                print(f"     → [{e['texte']}] = {e['type']} (conf: {e['confiance']:.2f})")


def main():
    parser = argparse.ArgumentParser(description="NER CamemBERT+LoRA pour manuscrits médiévaux")
    parser.add_argument("--mode", choices=["train", "predict", "all"], default="all")
    parser.add_argument("--input", type=Path, default=Path("data/normalise.json"))
    parser.add_argument("--output", type=Path, default=Path("data/entites.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ner"))
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    modele, tokeniseur = None, None

    if args.mode in ("train", "all"):
        modele, tokeniseur = mode_train(args)

    if args.mode in ("predict", "all"):
        mode_predict(args, modele, tokeniseur)


if __name__ == "__main__":
    main()