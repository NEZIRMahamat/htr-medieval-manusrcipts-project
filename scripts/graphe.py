"""
Script de construction du graphe de relations et export TEI-XML.

Usage :
    python scripts/graphe.py --input data/entites.json --output data/
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

from src.graphe.constructeur import (
    ConstructeurGraphe,
    exporter_tei_xml,
)


def main():
    parser = argparse.ArgumentParser(description="Graphe de relations + TEI-XML")
    parser.add_argument("--input", type=Path, default=Path("data/entites.json"))
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()

    # ── Chargement ──────────────────────────────────────────────────────────────
    print(f"📂 Chargement : {args.input}")
    data = json.loads(args.input.read_text(encoding="utf-8"))
    lignes = data.get("lignes", [])
    print(f"   {len(lignes)} lignes chargées")

    # ── Construction du graphe ──────────────────────────────────────────────────
    print("\n🔗 Construction du graphe de relations...")
    constructeur = ConstructeurGraphe(seuil_confiance=0.7)
    constructeur.traiter_corpus(lignes)

    # ── Statistiques ────────────────────────────────────────────────────────────
    stats = constructeur.statistiques()
    print(f"\n📊 Statistiques du graphe :")
    print(f"   Nœuds (entités)  : {stats['nb_noeuds']}")
    print(f"   Arêtes (relations) : {stats['nb_aretes']}")

    print(f"\n   Entités par type :")
    for type_e, count in stats["nb_entites_par_type"].items():
        print(f"   {type_e:10} : {count}")

    print(f"\n   Relations par type :")
    for type_r, count in stats["nb_relations_par_type"].items():
        print(f"   {type_r:15} : {count}")

    if stats["entites_les_plus_connectees"]:
        print(f"\n   Entités les plus connectées :")
        for item in stats["entites_les_plus_connectees"]:
            print(f"   {item['entite']:30} : degré {item['degre']}")

    # ── Export GraphML ──────────────────────────────────────────────────────────
    chemin_graphml = args.output / "graphe.graphml"
    constructeur.exporter_graphml(chemin_graphml)

    # ── Export TEI-XML ──────────────────────────────────────────────────────────
    print("\n📄 Export TEI-XML...")
    chemin_tei = args.output / "tei_output.xml"
    exporter_tei_xml(lignes, constructeur, chemin_tei)

    # ── Export JSON du graphe ───────────────────────────────────────────────────
    graphe_json = {
        "metadata": stats,
        "noeuds": [
            {
                "id": e.id,
                "texte": e.texte,
                "type": e.type,
                "occurrences": e.occurrences,
            }
            for e in constructeur.entites.values()
        ],
        "aretes": [
            {
                "source": r.source,
                "cible": r.cible,
                "type": r.type,
                "confiance": r.confiance,
                "contexte": r.contexte,
            }
            for r in constructeur.relations
        ],
    }

    chemin_json = args.output / "graphe.json"
    chemin_json.write_text(
        json.dumps(graphe_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"✅ Graphe JSON exporté → {chemin_json}")

    print(f"\n🎉 Pipeline NLP complet !")
    print(f"   Fichiers produits dans {args.output}/ :")
    print(f"   - normalise.json   (transcriptions normalisées)")
    print(f"   - entites.json     (entités NER)")
    print(f"   - pos.json         (annotations POS)")
    print(f"   - graphe.json      (graphe de relations)")
    print(f"   - graphe.graphml   (graphe format GraphML)")
    print(f"   - tei_output.xml   (export TEI-XML)")


if __name__ == "__main__":
    main()