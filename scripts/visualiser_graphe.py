"""
Visualisation du graphe de relations entre entités.

Usage :
    python scripts/visualiser_graphe.py --input data/graphe.json
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Couleurs par type d'entité
COULEURS = {
    "PER": "#4A90D9",   # bleu → personnes
    "LOC": "#E8A838",   # orange → lieux
    "ORG": "#7BC67E",   # vert → organisations
    "DATE": "#C67BB8",  # violet → dates
    "MISC": "#A0A0A0",  # gris → autre
}

# Couleurs par type de relation
COULEURS_ARETES = {
    "CO_PRESENCE": "#888888",
    "LIEU_DE": "#E8A838",
    "DESTINATION": "#4A90D9",
    "ORIGINE": "#E05C5C",
    "TITRE_DE": "#7BC67E",
}


def visualiser(chemin_json: Path, chemin_sortie: Path | None = None):
    """Charge et affiche le graphe de relations.

    Args:
        chemin_json: Chemin vers graphe.json
        chemin_sortie: Si fourni, sauvegarde l'image (PNG)
    """
    # Charger le graphe
    data = json.loads(chemin_json.read_text(encoding="utf-8"))
    noeuds = data.get("noeuds", [])
    aretes = data.get("aretes", [])

    if not noeuds:
        print("❌ Aucun nœud dans le graphe. Lancez d'abord le pipeline complet.")
        return

    # Construire le graphe NetworkX
    G = nx.DiGraph()

    for n in noeuds:
        G.add_node(
            n["id"],
            texte=n["texte"],
            type=n["type"],
            occurrences=n["occurrences"],
        )

    for a in aretes:
        G.add_edge(
            a["source"],
            a["cible"],
            type=a["type"],
            confiance=a["confiance"],
        )

    # ── Mise en page ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Position des nœuds
    if len(G.nodes) > 0:
        try:
            pos = nx.spring_layout(G, k=2, seed=42)
        except Exception:
            pos = nx.circular_layout(G)
    else:
        print("❌ Graphe vide.")
        return

    # Couleurs des nœuds selon leur type
    couleurs_noeuds = [
        COULEURS.get(G.nodes[n].get("type", "MISC"), "#A0A0A0")
        for n in G.nodes
    ]

    # Taille des nœuds selon les occurrences
    tailles = [
        300 + G.nodes[n].get("occurrences", 1) * 200
        for n in G.nodes
    ]

    # Couleurs des arêtes
    couleurs_aretes = [
        COULEURS_ARETES.get(G.edges[e].get("type", "CO_PRESENCE"), "#888888")
        for e in G.edges
    ]

    # ── Dessiner ────────────────────────────────────────────────────────────────

    # Arêtes
    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(
            G, pos,
            edge_color=couleurs_aretes,
            arrows=True,
            arrowsize=20,
            width=2,
            alpha=0.7,
            ax=ax,
            connectionstyle="arc3,rad=0.1",
        )

        # Labels des arêtes
        edge_labels = {
            (u, v): G.edges[u, v].get("type", "")
            for u, v in G.edges
        }
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=8,
            font_color="#CCCCCC",
            ax=ax,
        )

    # Nœuds
    nx.draw_networkx_nodes(
        G, pos,
        node_color=couleurs_noeuds,
        node_size=tailles,
        alpha=0.95,
        ax=ax,
    )

    # Labels des nœuds (texte de l'entité)
    labels = {n: G.nodes[n].get("texte", n) for n in G.nodes}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=10,
        font_color="white",
        font_weight="bold",
        ax=ax,
    )

    # ── Légende ─────────────────────────────────────────────────────────────────
    legendes = [
        mpatches.Patch(color=couleur, label=type_e)
        for type_e, couleur in COULEURS.items()
        if any(G.nodes[n].get("type") == type_e for n in G.nodes)
    ]
    ax.legend(
        handles=legendes,
        loc="upper left",
        facecolor="#2d2d44",
        labelcolor="white",
        fontsize=10,
    )

    # Titre
    nb_noeuds = G.number_of_nodes()
    nb_aretes = G.number_of_edges()
    ax.set_title(
        f"Graphe de relations — {nb_noeuds} entités, {nb_aretes} relations",
        color="white",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.axis("off")

    plt.tight_layout()

    if chemin_sortie:
        plt.savefig(str(chemin_sortie), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"✅ Graphe sauvegardé → {chemin_sortie}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualisation du graphe de relations")
    parser.add_argument("--input", type=Path, default=Path("data/graphe.json"))
    parser.add_argument("--output", type=Path, default=None,
                        help="Si fourni, sauvegarde en PNG au lieu d'afficher")
    args = parser.parse_args()

    print(f"📂 Chargement : {args.input}")
    visualiser(args.input, args.output)


if __name__ == "__main__":
    main()