"""
Annotation POS (Part-of-Speech) et lemmatisation avec Stanza.

Stanza est une bibliothèque NLP de Stanford qui supporte le français médiéval
via le modèle 'frm' (français médiéval) ou 'fr' (français moderne).

Ce module produit pour chaque token :
  - Le lemme (forme canonique)
  - La catégorie grammaticale (POS) : NOUN, VERB, ADJ, DET, PRON...
  - Les traits morphologiques : genre, nombre, temps, personne...

Références :
  - Stanza : Qi et al. (2020). Stanza: A Python NLP Library for Many Languages.
  - Modèle frm : entraîné sur l'SRCMF (Syntactic Reference Corpus of Medieval French)
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field


# ── Structures de données ──────────────────────────────────────────────────────


@dataclass
class TokenAnnote:
    """Token avec annotation POS et lemme.

    Attributes:
        texte: Forme de surface du token.
        lemme: Forme canonique (infinitif pour les verbes, singulier pour les noms).
        pos: Catégorie grammaticale universelle (UPOS).
        traits: Traits morphologiques (genre, nombre, temps...).
        indice: Position dans la phrase.
    """
    texte: str
    lemme: str
    pos: str
    traits: dict = field(default_factory=dict)
    indice: int = 0

    def to_dict(self) -> dict:
        return {
            "texte": self.texte,
            "lemme": self.lemme,
            "pos": self.pos,
            "traits": self.traits,
            "indice": self.indice,
        }


@dataclass
class PhraseAnnotee:
    """Phrase annotée avec POS et lemmes.

    Attributes:
        texte_original: Texte de la phrase.
        tokens: Liste de TokenAnnote.
    """
    texte_original: str
    tokens: list[TokenAnnote] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "texte_original": self.texte_original,
            "tokens": [t.to_dict() for t in self.tokens],
        }


# ── Descriptions des POS ──────────────────────────────────────────────────────

DESCRIPTIONS_POS = {
    "NOUN":  "Nom commun",
    "PROPN": "Nom propre",
    "VERB":  "Verbe",
    "AUX":   "Auxiliaire",
    "ADJ":   "Adjectif",
    "ADV":   "Adverbe",
    "DET":   "Déterminant",
    "PRON":  "Pronom",
    "ADP":   "Préposition",
    "CCONJ": "Conjonction de coordination",
    "SCONJ": "Conjonction de subordination",
    "PUNCT": "Ponctuation",
    "NUM":   "Nombre",
    "X":     "Autre",
}


# ── Annotateur POS ─────────────────────────────────────────────────────────────


class AnnotateurPOS:
    """Annotateur POS utilisant Stanza.

    Charge le modèle français (fr) ou français médiéval (frm) selon
    la disponibilité. Le modèle frm est spécialisé pour les manuscrits
    médiévaux mais nécessite un téléchargement séparé.

    Args:
        langue: Code de langue Stanza ('fr' ou 'frm').
        telecharger: Si True, télécharge le modèle si absent.

    Example:
        >>> annotateur = AnnotateurPOS(langue='fr')
        >>> resultats = annotateur.annoter_texte("le roi est mort")
    """

    def __init__(
        self,
        langue: str = "fr",
        telecharger: bool = True,
    ) -> None:
        self.langue = langue
        self._nlp = None
        self._charger_modele(telecharger)

    def _charger_modele(self, telecharger: bool) -> None:
        """Charge le modèle Stanza."""
        try:
            import stanza

            if telecharger:
                print(f"📥 Téléchargement du modèle Stanza '{self.langue}'...")
                stanza.download(self.langue, verbose=False)

            self._nlp = stanza.Pipeline(
                lang=self.langue,
                processors="tokenize,pos,lemma",
                verbose=False,
                tokenize_pretokenized=False,
            )
            print(f"✅ Modèle Stanza '{self.langue}' chargé !")

        except Exception as e:
            print(f"⚠️  Impossible de charger Stanza : {e}")
            print("   Utilisation du mode heuristique simplifié.")
            self._nlp = None

    def annoter_texte(self, texte: str) -> PhraseAnnotee:
        """Annote un texte avec POS et lemmes.

        Args:
            texte: Texte à annoter (une ligne normalisée).

        Returns:
            PhraseAnnotee avec tous les tokens annotés.

        Example:
            >>> phrase = annotateur.annoter_texte("le roi est parti de Paris")
            >>> for t in phrase.tokens:
            ...     print(f"{t.texte:15} {t.pos:8} {t.lemme}")
            le              DET      le
            roi             NOUN     roi
            est             AUX      être
            parti           VERB     partir
            de              ADP      de
            Paris           PROPN    Paris
        """
        if self._nlp is not None:
            return self._annoter_avec_stanza(texte)
        else:
            return self._annoter_heuristique(texte)

    def _annoter_avec_stanza(self, texte: str) -> PhraseAnnotee:
        """Annotation avec Stanza."""
        doc = self._nlp(texte)
        tokens = []
        indice = 0

        for phrase in doc.sentences:
            for token in phrase.words:
                # Parser les traits morphologiques
                traits = {}
                if token.feats:
                    for feat in token.feats.split("|"):
                        if "=" in feat:
                            cle, valeur = feat.split("=", 1)
                            traits[cle] = valeur

                tokens.append(TokenAnnote(
                    texte=token.text,
                    lemme=token.lemma or token.text,
                    pos=token.upos or "X",
                    traits=traits,
                    indice=indice,
                ))
                indice += 1

        return PhraseAnnotee(texte_original=texte, tokens=tokens)

    def _annoter_heuristique(self, texte: str) -> PhraseAnnotee:
        """Annotation heuristique simplifiée (fallback sans Stanza).

        Règles simples basées sur des listes de mots médiévaux fréquents.
        """
        DETERMINANTS = {"le", "la", "les", "un", "une", "des", "du", "de", "li", "lo"}
        PREPOSITIONS = {"de", "en", "por", "par", "vers", "a", "au", "aux", "sur"}
        CONJONCTIONS = {"et", "ou", "mais", "que", "qe", "mes", "ne", "ni"}
        PRONOMS = {"il", "ils", "elle", "elles", "nous", "vous", "on", "lui",
                   "icil", "icele", "cil", "cele"}
        AUXILIAIRES = {"est", "sont", "fu", "furent", "avoit", "avoir", "estre"}

        mots = texte.split()
        tokens = []

        for i, mot in enumerate(mots):
            mot_lower = mot.lower().strip(".,;:!?()[]")

            if mot_lower in DETERMINANTS:
                pos, lemme = "DET", mot_lower
            elif mot_lower in PREPOSITIONS:
                pos, lemme = "ADP", mot_lower
            elif mot_lower in CONJONCTIONS:
                pos, lemme = "CCONJ", mot_lower
            elif mot_lower in PRONOMS:
                pos, lemme = "PRON", mot_lower
            elif mot_lower in AUXILIAIRES:
                pos, lemme = "AUX", "être"
            elif mot[0].isupper() and i > 0:
                pos, lemme = "PROPN", mot
            elif mot_lower.endswith(("er", "ir", "re", "oir")):
                pos, lemme = "VERB", mot_lower
            elif mot_lower.endswith(("ment", "ment")):
                pos, lemme = "ADV", mot_lower
            else:
                pos, lemme = "NOUN", mot_lower

            tokens.append(TokenAnnote(
                texte=mot,
                lemme=lemme,
                pos=pos,
                traits={},
                indice=i,
            ))

        return PhraseAnnotee(texte_original=texte, tokens=tokens)

    def annoter_corpus(
        self,
        lignes: list[dict],
    ) -> list[dict]:
        """Annote toutes les lignes d'un corpus.

        Args:
            lignes: Liste de dicts avec "texte_normalise".

        Returns:
            Liste de dicts avec "annotation_pos" ajouté.

        Example:
            >>> resultats = annotateur.annoter_corpus(lignes_normalisees)
        """
        resultats = []

        for i, ligne in enumerate(lignes):
            texte = ligne.get("texte_normalise", ligne.get("transcription", ""))

            if texte.strip():
                phrase = self.annoter_texte(texte)
                annotation = phrase.to_dict()
            else:
                annotation = {"texte_original": texte, "tokens": []}

            resultats.append({
                **ligne,
                "annotation_pos": annotation,
            })

            if (i + 1) % 5 == 0:
                print(f"   {i+1}/{len(lignes)} lignes annotées...")

        return resultats


# ── Statistiques POS ───────────────────────────────────────────────────────────


def statistiques_pos(lignes_annotees: list[dict]) -> dict:
    """Calcule les statistiques POS sur le corpus annoté.

    Args:
        lignes_annotees: Sortie de annoter_corpus().

    Returns:
        Dictionnaire de statistiques.

    Example:
        >>> stats = statistiques_pos(resultats)
        >>> stats["distribution_pos"]
        {"NOUN": 45, "VERB": 32, "DET": 28, ...}
    """
    compteur_pos: dict[str, int] = {}
    compteur_lemmes: dict[str, int] = {}
    nb_tokens_total = 0

    for ligne in lignes_annotees:
        tokens = ligne.get("annotation_pos", {}).get("tokens", [])
        for token in tokens:
            pos = token.get("pos", "X")
            lemme = token.get("lemme", "")
            compteur_pos[pos] = compteur_pos.get(pos, 0) + 1
            compteur_lemmes[lemme] = compteur_lemmes.get(lemme, 0) + 1
            nb_tokens_total += 1

    # Top 10 lemmes les plus fréquents (hors mots grammaticaux)
    pos_grammaticaux = {"DET", "ADP", "CCONJ", "SCONJ", "PUNCT", "PRON"}
    lemmes_lexicaux = {
        l: c for l, c in compteur_lemmes.items()
        if l and len(l) > 2
    }
    top_lemmes = sorted(lemmes_lexicaux.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "nb_tokens_total": nb_tokens_total,
        "nb_lignes": len(lignes_annotees),
        "distribution_pos": compteur_pos,
        "top_10_lemmes": top_lemmes,
        "descriptions_pos": {
            pos: DESCRIPTIONS_POS.get(pos, pos)
            for pos in compteur_pos
        },
    }