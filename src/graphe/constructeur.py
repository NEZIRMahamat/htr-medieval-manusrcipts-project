"""
Construction du graphe de relations entre entités nommées.

Ce module :
  1. Extrait les relations entre entités par règles simples
  2. Construit un graphe NetworkX (nœuds = entités, arêtes = relations)
  3. Exporte en TEI-XML (standard pour les humanités numériques)

Types de relations détectées par règles :
  - LIEU_DE (personne → lieu)       : "le roi Phelippe de France"
  - TITRE_DE (titre → personne)     : "monseigneur Jehan duc de Berry"
  - CO_PRESENCE (personne + personne dans la même ligne)
  - ORIGINE (personne → lieu)       : "parti de Paris"

Références :
  - TEI Consortium (2023). TEI P5: Guidelines for Electronic Text Encoding.
  - NetworkX : Hagberg et al. (2008). Exploring network structure with NetworkX.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

import networkx as nx


# ── Structures de données ──────────────────────────────────────────────────────


@dataclass
class Entite:
    """Entité nommée dans le graphe.

    Attributes:
        id: Identifiant unique (ex: "PER_Phelippe").
        texte: Forme de surface.
        type: PER, LOC, ORG, DATE, TITLE.
        occurrences: Nombre d'occurrences dans le corpus.
        pages: Pages où l'entité apparaît.
    """
    id: str
    texte: str
    type: str
    occurrences: int = 1
    pages: list[str] = field(default_factory=list)


@dataclass
class Relation:
    """Relation entre deux entités.

    Attributes:
        source: ID de l'entité source.
        cible: ID de l'entité cible.
        type: Type de relation.
        confiance: Score de confiance [0, 1].
        contexte: Texte de la ligne où la relation a été détectée.
    """
    source: str
    cible: str
    type: str
    confiance: float = 1.0
    contexte: str = ""


# ── Types de relations ────────────────────────────────────────────────────────

TYPES_RELATIONS = {
    "CO_PRESENCE": "co-présence dans le même contexte",
    "LIEU_DE": "lieu associé à une personne",
    "TITRE_DE": "titre associé à une personne",
    "ORIGINE": "lieu d'origine ou de départ",
    "DESTINATION": "lieu de destination",
}

# Mots-clés indiquant une relation de lieu
MOTS_ORIGINE = {"de", "parti", "venu", "issu", "natif"}
MOTS_DESTINATION = {"vers", "a", "au", "en", "pour"}
MOTS_TITRE = {"duc", "comte", "roi", "roy", "sire", "seigneur", "monseigneur", "mgr"}


# ── Extraction des relations ───────────────────────────────────────────────────


def extraire_relations_ligne(
    texte: str,
    entites: list[dict],
    id_page: str = "",
) -> list[Relation]:
    """Extrait les relations entre entités d'une ligne par règles simples.

    Args:
        texte: Texte normalisé de la ligne.
        entites: Entités détectées dans cette ligne.
        id_page: Identifiant de la page source.

    Returns:
        Liste de Relations détectées.

    Example:
        >>> entites = [{"texte": "Phelippe", "type": "PER"},
        ...            {"texte": "France", "type": "LOC"}]
        >>> relations = extraire_relations_ligne("le roi Phelippe de France", entites)
        >>> relations[0].type
        "CO_PRESENCE"
    """
    relations = []
    mots = texte.lower().split()

    personnes = [e for e in entites if e["type"] in ("PER",)]
    lieux = [e for e in entites if e["type"] in ("LOC",)]
    titres = [e for e in entites if e["type"] in ("TITLE", "ORG")]

    # ── Règle 1 : CO_PRESENCE ────────────────────────────────────────────────
    # Deux personnes dans la même ligne → relation de co-présence
    for i in range(len(personnes)):
        for j in range(i + 1, len(personnes)):
            relations.append(Relation(
                source=f"PER_{personnes[i]['texte']}",
                cible=f"PER_{personnes[j]['texte']}",
                type="CO_PRESENCE",
                confiance=0.7,
                contexte=texte[:80],
            ))

    # ── Règle 2 : LIEU_DE ────────────────────────────────────────────────────
    # Personne + "de" + lieu → LIEU_DE
    for personne in personnes:
        for lieu in lieux:
            # Vérifier si "de" apparaît entre les deux entités dans le texte
            texte_lower = texte.lower()
            pos_per = texte_lower.find(personne["texte"].lower())
            pos_loc = texte_lower.find(lieu["texte"].lower())

            if pos_per >= 0 and pos_loc >= 0:
                segment = texte_lower[min(pos_per, pos_loc):max(pos_per, pos_loc)]
                if " de " in segment or segment.startswith("de "):
                    relations.append(Relation(
                        source=f"PER_{personne['texte']}",
                        cible=f"LOC_{lieu['texte']}",
                        type="LIEU_DE",
                        confiance=0.85,
                        contexte=texte[:80],
                    ))

    # ── Règle 3 : DESTINATION / ORIGINE ─────────────────────────────────────
    # "vers [lieu]" → DESTINATION, "de [lieu]" / "parti de [lieu]" → ORIGINE
    for lieu in lieux:
        texte_lower = texte.lower()
        pos_loc = texte_lower.find(lieu["texte"].lower())

        if pos_loc > 0:
            # Regarder les 2 mots avant le lieu
            avant = texte_lower[:pos_loc].strip().split()[-2:]

            if any(m in MOTS_DESTINATION for m in avant):
                # Trouver la personne la plus proche
                if personnes:
                    relations.append(Relation(
                        source=f"PER_{personnes[0]['texte']}",
                        cible=f"LOC_{lieu['texte']}",
                        type="DESTINATION",
                        confiance=0.8,
                        contexte=texte[:80],
                    ))

            elif any(m in MOTS_ORIGINE for m in avant):
                if personnes:
                    relations.append(Relation(
                        source=f"PER_{personnes[0]['texte']}",
                        cible=f"LOC_{lieu['texte']}",
                        type="ORIGINE",
                        confiance=0.8,
                        contexte=texte[:80],
                    ))

    return relations


# ── Construction du graphe ─────────────────────────────────────────────────────


class ConstructeurGraphe:
    """Construit un graphe de relations à partir des entités annotées.

    Args:
        seuil_confiance: Score minimum pour inclure une relation.

    Example:
        >>> constructeur = ConstructeurGraphe()
        >>> constructeur.traiter_corpus(lignes_avec_entites)
        >>> constructeur.exporter_graphml("data/graphe.graphml")
    """

    def __init__(self, seuil_confiance: float = 0.7) -> None:
        self.graphe = nx.DiGraph()
        self.seuil_confiance = seuil_confiance
        self.entites: dict[str, Entite] = {}
        self.relations: list[Relation] = []

    def ajouter_entite(self, texte: str, type_entite: str, page: str = "") -> str:
        """Ajoute une entité au graphe ou incrémente son compteur.

        Args:
            texte: Forme de surface de l'entité.
            type_entite: PER, LOC, ORG, DATE, TITLE.
            page: Page source.

        Returns:
            ID de l'entité.
        """
        id_entite = f"{type_entite}_{texte}"

        if id_entite not in self.entites:
            self.entites[id_entite] = Entite(
                id=id_entite,
                texte=texte,
                type=type_entite,
                occurrences=1,
                pages=[page] if page else [],
            )
            # Ajouter le nœud au graphe
            self.graphe.add_node(
                id_entite,
                texte=texte,
                type=type_entite,
                occurrences=1,
            )
        else:
            self.entites[id_entite].occurrences += 1
            self.graphe.nodes[id_entite]["occurrences"] += 1
            if page and page not in self.entites[id_entite].pages:
                self.entites[id_entite].pages.append(page)

        return id_entite

    def ajouter_relation(self, relation: Relation) -> None:
        """Ajoute une relation au graphe.

        Args:
            relation: Relation à ajouter.
        """
        if relation.confiance < self.seuil_confiance:
            return

        self.relations.append(relation)

        # Ajouter l'arête au graphe
        if self.graphe.has_edge(relation.source, relation.cible):
            # Incrémenter le poids si la relation existe déjà
            self.graphe[relation.source][relation.cible]["poids"] += 1
        else:
            self.graphe.add_edge(
                relation.source,
                relation.cible,
                type=relation.type,
                confiance=relation.confiance,
                contexte=relation.contexte,
                poids=1,
            )

    def traiter_corpus(self, lignes: list[dict]) -> None:
        """Traite toutes les lignes annotées et construit le graphe.

        Args:
            lignes: Sortie du module NER avec "entites" et "id_page".
        """
        for ligne in lignes:
            entites = ligne.get("entites", [])
            id_page = ligne.get("id_page", "")
            texte = ligne.get("texte_normalise", ligne.get("transcription", ""))

            # Ajouter les entités
            for e in entites:
                self.ajouter_entite(e["texte"], e["type"], id_page)

            # Extraire et ajouter les relations
            if len(entites) >= 2:
                relations = extraire_relations_ligne(texte, entites, id_page)
                for relation in relations:
                    # Vérifier que les nœuds existent
                    if (relation.source in self.graphe.nodes and
                            relation.cible in self.graphe.nodes):
                        self.ajouter_relation(relation)

    def statistiques(self) -> dict:
        """Retourne les statistiques du graphe.

        Returns:
            Dictionnaire de statistiques.
        """
        return {
            "nb_noeuds": self.graphe.number_of_nodes(),
            "nb_aretes": self.graphe.number_of_edges(),
            "nb_entites_par_type": {
                type_e: sum(1 for e in self.entites.values() if e.type == type_e)
                for type_e in set(e.type for e in self.entites.values())
            },
            "nb_relations_par_type": {
                type_r: sum(1 for r in self.relations if r.type == type_r)
                for type_r in set(r.type for r in self.relations)
            },
            "entites_les_plus_connectees": [
                {"entite": n, "degre": d}
                for n, d in sorted(
                    self.graphe.degree(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]
            ],
        }

    def exporter_graphml(self, chemin: str | Path) -> None:
        """Exporte le graphe au format GraphML.

        Args:
            chemin: Chemin du fichier .graphml.
        """
        chemin = Path(chemin)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(self.graphe, str(chemin))
        print(f"✅ Graphe exporté → {chemin}")


# ── Export TEI-XML ─────────────────────────────────────────────────────────────


def exporter_tei_xml(
    lignes: list[dict],
    constructeur: ConstructeurGraphe,
    chemin_sortie: str | Path,
    titre: str = "Transcriptions HTR médiévales",
) -> None:
    """Exporte les transcriptions annotées en TEI-XML.

    TEI-XML est le standard des humanités numériques pour l'encodage
    de textes avec annotations linguistiques et entités nommées.

    Args:
        lignes: Lignes annotées (avec entités et POS).
        constructeur: Graphe de relations construit.
        chemin_sortie: Chemin du fichier .xml.
        titre: Titre du document TEI.

    Example:
        >>> exporter_tei_xml(lignes, constructeur, "data/tei_output.xml")
    """
    from lxml import etree

    TEI_NS = "http://www.tei-c.org/ns/1.0"
    XML_NS = "http://www.w3.org/XML/1998/namespace"

    # ── Racine TEI ────────────────────────────────────────────────────────────
    tei = etree.Element(f"{{{TEI_NS}}}TEI", nsmap={None: TEI_NS})

    # ── En-tête (teiHeader) ───────────────────────────────────────────────────
    header = etree.SubElement(tei, f"{{{TEI_NS}}}teiHeader")
    file_desc = etree.SubElement(header, f"{{{TEI_NS}}}fileDesc")

    title_stmt = etree.SubElement(file_desc, f"{{{TEI_NS}}}titleStmt")
    title_elem = etree.SubElement(title_stmt, f"{{{TEI_NS}}}title")
    title_elem.text = titre

    pub_stmt = etree.SubElement(file_desc, f"{{{TEI_NS}}}publicationStmt")
    p_pub = etree.SubElement(pub_stmt, f"{{{TEI_NS}}}p")
    p_pub.text = "Pipeline HTR NLP - Master Data/IA HETIC 2026"

    source_desc = etree.SubElement(file_desc, f"{{{TEI_NS}}}sourceDesc")
    p_source = etree.SubElement(source_desc, f"{{{TEI_NS}}}p")
    p_source.text = "Manuscrits médiévaux numérisés - Gallica BnF"

    # ── Corps du texte (body) ────────────────────────────────────────────────
    text_elem = etree.SubElement(tei, f"{{{TEI_NS}}}text")
    body = etree.SubElement(text_elem, f"{{{TEI_NS}}}body")
    div = etree.SubElement(body, f"{{{TEI_NS}}}div")

    # Regrouper les lignes par page
    pages: dict[str, list[dict]] = defaultdict(list)
    for ligne in lignes:
        page_id = ligne.get("id_page", "page_inconnue")
        pages[page_id].append(ligne)

    for page_id, lignes_page in pages.items():
        # Élément de page
        pb = etree.SubElement(div, f"{{{TEI_NS}}}pb")
        pb.set(f"{{{XML_NS}}}id", page_id)

        for ligne in lignes_page:
            texte = ligne.get("texte_normalise", ligne.get("transcription", ""))
            entites = ligne.get("entites", [])
            id_ligne = ligne.get("id_ligne", "")

            # Élément de ligne
            lb = etree.SubElement(div, f"{{{TEI_NS}}}lb")
            lb.set(f"{{{XML_NS}}}id", id_ligne)

            if not entites:
                # Ligne sans entités : texte simple
                p_elem = etree.SubElement(div, f"{{{TEI_NS}}}p")
                p_elem.text = texte
            else:
                # Ligne avec entités : annoter les entités
                p_elem = etree.SubElement(div, f"{{{TEI_NS}}}p")
                texte_restant = texte

                for entite in entites:
                    texte_entite = entite["texte"]
                    pos_entite = texte_restant.find(texte_entite)

                    if pos_entite < 0:
                        continue

                    # Texte avant l'entité
                    if pos_entite > 0:
                        span_avant = etree.SubElement(p_elem, f"{{{TEI_NS}}}seg")
                        span_avant.text = texte_restant[:pos_entite]

                    # L'entité elle-même
                    type_tei = {
                        "PER": "persName",
                        "LOC": "placeName",
                        "ORG": "orgName",
                        "DATE": "date",
                        "TITLE": "roleName",
                        "MISC": "name",
                    }.get(entite["type"], "name")

                    entite_elem = etree.SubElement(p_elem, f"{{{TEI_NS}}}{type_tei}")
                    entite_elem.text = texte_entite
                    entite_elem.set("type", entite["type"])

                    texte_restant = texte_restant[pos_entite + len(texte_entite):]

                # Texte restant après la dernière entité
                if texte_restant:
                    span_fin = etree.SubElement(p_elem, f"{{{TEI_NS}}}seg")
                    span_fin.text = texte_restant

    # ── Export ────────────────────────────────────────────────────────────────
    chemin_sortie = Path(chemin_sortie)
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

    arbre = etree.ElementTree(tei)
    arbre.write(
        str(chemin_sortie),
        pretty_print=True,
        xml_declaration=True,
        encoding="UTF-8",
    )
    print(f"✅ TEI-XML exporté → {chemin_sortie}")