"""
Script d'annotation POS et lemmatisation sur les transcriptions normalisées.

Usage :
    python scripts/pos.py --input data/normalise.json --output data/pos.json
    python scripts/pos.py --input data/normalise.json --output data/pos.json --langue frm
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

from src.pos.annotateur import AnnotateurPOS, statistiques_pos, DESCRIPTIONS_POS


def main():
    parser = argparse.ArgumentParser(description="Annotation POS avec Stanza")
    parser.add_argument("--input", type=Path, default=Path("data/normalise.json"))
    parser.add_argument("--output", type=Path, default=Path("data/pos.json"))
    parser.add_argument(
        "--langue",
        type=str,
        default="fr",
        choices=["fr", "frm"],
        help="fr=français moderne, frm=français médiéval (meilleur mais plus rare)",
    )
    args = parser.parse_args()

    # ── Chargement ──────────────────────────────────────────────────────────────
    print(f"📂 Chargement : {args.input}")
    data = json.loads(args.input.read_text(encoding="utf-8"))
    lignes = data.get("lignes", [])
    print(f"   {len(lignes)} lignes à annoter")

    # ── Annotation POS ──────────────────────────────────────────────────────────
    annotateur = AnnotateurPOS(langue=args.langue, telecharger=True)

    print("\n🔍 Annotation POS en cours...")
    resultats = annotateur.annoter_corpus(lignes)

    # ── Statistiques ────────────────────────────────────────────────────────────
    stats = statistiques_pos(resultats)

    print(f"\n📊 Statistiques POS :")
    print(f"   Tokens totaux : {stats['nb_tokens_total']}")
    print(f"\n   Distribution des catégories grammaticales :")
    for pos, count in sorted(stats["distribution_pos"].items(), key=lambda x: x[1], reverse=True):
        desc = DESCRIPTIONS_POS.get(pos, pos)
        pct = count / stats["nb_tokens_total"] * 100
        print(f"   {pos:8} ({desc:30}) : {count:4} ({pct:.1f}%)")

    print(f"\n   Top 10 lemmes les plus fréquents :")
    for lemme, count in stats["top_10_lemmes"]:
        print(f"   '{lemme}' → {count} fois")

    # ── Exemples qualitatifs ────────────────────────────────────────────────────
    print(f"\n📖 Exemple d'annotation :")
    for r in resultats[:3]:
        tokens = r.get("annotation_pos", {}).get("tokens", [])
        if tokens:
            print(f"\n   Texte : {r.get('texte_normalise', '')[:60]}")
            print(f"   {'Token':<15} {'POS':<8} {'Lemme':<15} Traits")
            print(f"   {'-'*55}")
            for t in tokens[:8]:
                traits_str = " ".join(f"{k}={v}" for k, v in t.get("traits", {}).items())
                print(f"   {t['texte']:<15} {t['pos']:<8} {t['lemme']:<15} {traits_str}")

    # ── Export ──────────────────────────────────────────────────────────────────
    sortie = {
        "metadata": {
            "langue_stanza": args.langue,
            "nb_lignes": len(resultats),
            "statistiques": stats,
        },
        "lignes": resultats,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(sortie, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n✅ Annotations POS exportées → {args.output}")


if __name__ == "__main__":
    main()