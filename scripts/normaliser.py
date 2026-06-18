"""
Script de normalisation orthographique des transcriptions HTR.

Prend en entrée le JSON produit par le pipeline CV (data contract)
et produit un JSON normalisé pour le module NLP.

Usage :
    python scripts/normaliser.py --input dataset_nlp/transcriptions.json --output dataset_nlp/normalise.json
    python scripts/normaliser.py --input dataset_nlp/transcriptions.json --output dataset_nlp/normalise.json --evaluer
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

from src.normalisation.normaliseur import (
    NormaliseurRegles,
    statistiques_normalisation,
    exporter_json_normalise,
    evaluer_normalisation,
)


def main():
    parser = argparse.ArgumentParser(description="Normalisation orthographique HTR")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Fichier JSON du data contract HTR (sortie du pipeline CV)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_nlp/normalise.json"),
        help="Fichier JSON de sortie normalisé",
    )
    parser.add_argument(
        "--abreviations-custom",
        type=Path,
        default=None,
        help="Fichier JSON d'abréviations supplémentaires {abrév: expansion}",
    )
    parser.add_argument(
        "--evaluer",
        action="store_true",
        help="Évaluer l'impact sur le CER (nécessite des références dans le JSON)",
    )
    parser.add_argument(
        "--seuil-review",
        type=float,
        default=0.6,
        help="Seuil de confiance sous lequel une ligne est marquée needs_review",
    )
    args = parser.parse_args()

    # ── Chargement ──────────────────────────────────────────────────────────────
    print(f"📂 Chargement : {args.input}")
    data = json.loads(args.input.read_text(encoding="utf-8"))

    # Charger abréviations custom si fournies
    abrev_custom = None
    if args.abreviations_custom and args.abreviations_custom.exists():
        abrev_custom = json.loads(
            args.abreviations_custom.read_text(encoding="utf-8")
        )
        print(f"  ✅ {len(abrev_custom)} abréviations custom chargées")

    # ── Normalisation ───────────────────────────────────────────────────────────
    normaliseur = NormaliseurRegles(
        abreviations_custom=abrev_custom,
        activer_nasales=True,
        activer_regex=True,
    )

    toutes_lignes = []
    for page in data.get("pages", []):
        lignes_normalisees = normaliseur.normaliser_page(
            page.get("lignes", []),
            seuil_confiance_review=args.seuil_review,
        )
        toutes_lignes.extend(lignes_normalisees)

    print(f"  ✅ {len(toutes_lignes)} lignes normalisées")

    # ── Statistiques ────────────────────────────────────────────────────────────
    stats = statistiques_normalisation(toutes_lignes)

    print(f"\n📊 Statistiques de normalisation :")
    print(f"   Lignes modifiées    : {stats['nb_lignes_modifiees']} / {stats['nb_total_lignes']} ({stats['taux_modification']*100:.1f}%)")
    print(f"   Needs review        : {stats['nb_needs_review']} ({stats['taux_needs_review']*100:.1f}%)")
    print(f"   Total modifications : {stats['nb_total_modifications']}")
    print(f"\n   Types de modifications :")
    for type_mod, count in stats["modifications_par_type"].items():
        print(f"     - {type_mod}: {count}")
    print(f"\n   Top abréviations résolues :")
    for abrev, count in stats["top_10_abreviations"]:
        print(f"     - '{abrev}' → {count} fois")

    # ── Évaluation CER (si références disponibles) ─────────────────────────────
    if args.evaluer:
        paires_avant = []
        paires_apres = []

        for page in data.get("pages", []):
            for ligne_orig, ligne_norm in zip(
                page.get("lignes", []), toutes_lignes
            ):
                ref = ligne_orig.get("reference", "")
                if ref:
                    paires_avant.append((ref, ligne_orig.get("transcription", "")))
                    paires_apres.append((ref, ligne_norm.texte_normalise))

        if paires_avant:
            eval_stats = evaluer_normalisation(paires_avant, paires_apres)
            print(f"\n📈 Impact sur le CER :")
            print(f"   CER avant normalisation : {eval_stats['cer_avant']*100:.2f}%")
            print(f"   CER après normalisation : {eval_stats['cer_apres']*100:.2f}%")
            print(f"   Amélioration absolue    : -{eval_stats['amelioration_absolue']*100:.2f} points")
            print(f"   Amélioration relative   : -{eval_stats['amelioration_relative']*100:.1f}%")
            stats["evaluation_cer"] = eval_stats
        else:
            print("\n⚠️  Pas de références disponibles pour évaluer le CER.")

    # ── Export ──────────────────────────────────────────────────────────────────
    exporter_json_normalise(toutes_lignes, args.output, stats)


if __name__ == "__main__":
    main()
