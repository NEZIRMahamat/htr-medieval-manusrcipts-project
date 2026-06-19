"""
Extraction d'entités nommées avec un modèle NER français pré-entraîné.

Modèle utilisé : Jean-Baptiste/roberta-large-ner-french
  - Entraîné sur du français moderne
  - Détecte : PER, LOC, ORG, MISC
  - Fonctionne bien sur le vieux français normalisé

Usage :
    python scripts/ner_inference.py --input data/normalise.json --output data/entites.json
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from transformers import pipeline


MODELE_NER = "dslim/bert-base-NER"


def charger_pipeline_ner():
    print(f"📥 Chargement du modèle NER : {MODELE_NER}")
    print("   (téléchargement ~1.4Go la première fois, patience...)")
    ner_pipeline = pipeline(
        "ner",
        model=MODELE_NER,
        aggregation_strategy="simple",
    )
    print("✅ Modèle chargé !")
    return ner_pipeline


def extraire_entites_ligne(texte: str, ner_pipeline) -> list[dict]:
    if not texte.strip():
        return []
    try:
        resultats = ner_pipeline(texte)
    except Exception:
        return []
    return [
        {
            "texte": r["word"].strip(),
            "type": r["entity_group"],
            "confiance": round(float(r["score"]), 4),
        }
        for r in resultats
    ]


def annoter_transcriptions(lignes, ner_pipeline, seuil_confiance=0.7):
    resultats = []
    for i, ligne in enumerate(lignes):
        texte = ligne.get("texte_normalise", ligne.get("transcription", ""))
        entites = [
            e for e in extraire_entites_ligne(texte, ner_pipeline)
            if e["confiance"] >= seuil_confiance
        ]
        resultats.append({**ligne, "entites": entites, "nb_entites": len(entites)})
        if (i + 1) % 5 == 0:
            print(f"   {i+1}/{len(lignes)} lignes traitées...")
    return resultats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/normalise.json"))
    parser.add_argument("--output", type=Path, default=Path("data/entites.json"))
    parser.add_argument("--seuil", type=float, default=0.7)
    args = parser.parse_args()

    ner_pipeline = charger_pipeline_ner()

    print(f"\n📂 Chargement : {args.input}")
    data = json.loads(args.input.read_text(encoding="utf-8"))
    lignes = data.get("lignes", [])
    print(f"   {len(lignes)} lignes à annoter")

    print("\n🔍 Extraction des entités...")
    resultats = annoter_transcriptions(lignes, ner_pipeline, args.seuil)

    toutes_entites = [e for r in resultats for e in r.get("entites", [])]
    stats_types: dict[str, int] = {}
    for e in toutes_entites:
        stats_types[e["type"]] = stats_types.get(e["type"], 0) + 1

    print(f"\n📊 Résultats :")
    print(f"   Entités extraites : {len(toutes_entites)}")
    for type_ent, count in sorted(stats_types.items()):
        print(f"   {type_ent:10} : {count}")

    print("\n📖 Exemples de détections :")
    for r in resultats:
        if r.get("entites"):
            print(f"\n   Texte : {r.get('texte_normalise', '')[:70]}")
            for e in r["entites"]:
                print(f"     → [{e['texte']}] = {e['type']} ({e['confiance']:.2f})")

    sortie = {
        "metadata": {
            "modele_ner": MODELE_NER,
            "nb_lignes": len(resultats),
            "nb_entites_total": len(toutes_entites),
            "entites_par_type": stats_types,
            "seuil_confiance": args.seuil,
        },
        "lignes": resultats,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sortie, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Résultats exportés → {args.output}")


if __name__ == "__main__":
    main()