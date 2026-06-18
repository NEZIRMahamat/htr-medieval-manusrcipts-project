"""
Pipeline de normalisation orthographique pour les transcriptions HTR médiévales.

Trois niveaux de normalisation (activables indépendamment) :
  1. Règles déterministes : table d'abréviations + regex
  2. Filtrage par confiance : on ne normalise que les lignes incertaines
  3. [Optionnel] CamemBERT MLM : correction des résidus par modèle de langue

Références :
  - Pinche et al. (2022). Normalizing Medieval French Texts.
  - Clerice et al. (2023). CATMuS Medieval.
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from dataclasses import dataclass, field

from src.normalisation.abreviations import (
    ABREVIATIONS_DIRECTES,
    NASALES_TILDE,
    PATTERNS_REGEX,
)


# ── Structures de données ──────────────────────────────────────────────────────


@dataclass
class LigneNormalisee:
    """Résultat de la normalisation d'une ligne de texte.

    Attributes:
        id_ligne: Identifiant de la ligne dans la page.
        texte_original: Transcription brute du modèle HTR.
        texte_normalise: Transcription après normalisation.
        modifications: Liste des modifications appliquées.
        confiance_htr: Score de confiance HTR original [0, 1].
        needs_review: True si la ligne nécessite une vérification humaine.
    """

    id_ligne: str
    texte_original: str
    texte_normalise: str
    modifications: list[dict] = field(default_factory=list)
    confiance_htr: float = 1.0
    needs_review: bool = False

    @property
    def a_ete_modifie(self) -> bool:
        """True si la normalisation a changé le texte."""
        return self.texte_original != self.texte_normalise

    def to_dict(self) -> dict:
        """Sérialise en dictionnaire JSON-compatible."""
        return {
            "id_ligne": self.id_ligne,
            "texte_original": self.texte_original,
            "texte_normalise": self.texte_normalise,
            "modifications": self.modifications,
            "confiance_htr": self.confiance_htr,
            "needs_review": self.needs_review,
            "modifie": self.a_ete_modifie,
        }


# ── Normaliseur par règles ─────────────────────────────────────────────────────


class NormaliseurRegles:
    """Normalisation déterministe par table d'abréviations et expressions régulières.

    C'est la première couche du pipeline. Elle est rapide, explicable et
    entièrement reproductible — contrairement aux approches par modèle de langue.

    Args:
        abreviations_custom: Dictionnaire optionnel d'abréviations supplémentaires
                             (pour étendre la table de base).
        activer_nasales: Si True, expande les voyelles avec tilde/macron.
        activer_regex: Si True, applique les patterns regex.
        casse_sensible: Si True, respecte la casse pour les abréviations directes.

    Example:
        >>> norm = NormaliseurRegles()
        >>> norm.normaliser_ligne("le roi & ses vassaux sōt venus")
        "le roi et ses vassaux sont venus"
    """

    def __init__(
        self,
        abreviations_custom: dict[str, str] | None = None,
        activer_nasales: bool = True,
        activer_regex: bool = True,
        casse_sensible: bool = False,
    ) -> None:
        self.abreviations = dict(ABREVIATIONS_DIRECTES)
        if abreviations_custom:
            self.abreviations.update(abreviations_custom)

        self.nasales = NASALES_TILDE
        self.patterns_regex = PATTERNS_REGEX
        self.activer_nasales = activer_nasales
        self.activer_regex = activer_regex
        self.casse_sensible = casse_sensible

        # Compiler les patterns regex une seule fois
        self._patterns_compiles = [
            (re.compile(pattern), remplacement)
            for pattern, remplacement in self.patterns_regex
        ]

    def normaliser_ligne(self, texte: str) -> tuple[str, list[dict]]:
        """Normalise une ligne de texte et retourne les modifications appliquées.

        Args:
            texte: Texte brut issu du modèle HTR.

        Returns:
            Tuple (texte_normalise, liste_modifications) où chaque modification
            est un dict {"type", "avant", "apres", "position"}.

        Example:
            >>> norm = NormaliseurRegles()
            >>> texte, mods = norm.normaliser_ligne("qe le roy & la royne sōt partiz")
            >>> texte
            "que le roy et la royne sont partiz"
            >>> len(mods)
            3
        """
        modifications = []
        resultat = texte

        # Étape 1 : Nasales (voyelles avec tilde/macron)
        if self.activer_nasales:
            for char_abrege, expansion in self.nasales.items():
                if char_abrege in resultat:
                    avant = resultat
                    resultat = resultat.replace(char_abrege, expansion)
                    if resultat != avant:
                        modifications.append({
                            "type": "nasale",
                            "avant": char_abrege,
                            "apres": expansion,
                        })

        # Étape 2 : Abréviations directes (mots entiers)
        mots = resultat.split()
        mots_normalises = []
        for mot in mots:
            # Enlever la ponctuation pour la comparaison
            mot_clean = mot.strip(".,;:!?()[]")
            ponctuation = mot[len(mot_clean):]

            cle = mot_clean if self.casse_sensible else mot_clean.lower()

            if cle in self.abreviations:
                expansion = self.abreviations[cle]
                # Préserver la casse si le mot original est en majuscule
                if mot_clean and mot_clean[0].isupper():
                    expansion = expansion.capitalize()
                modifications.append({
                    "type": "abreviation",
                    "avant": mot_clean,
                    "apres": expansion,
                })
                mots_normalises.append(expansion + ponctuation)
            else:
                mots_normalises.append(mot)

        resultat = " ".join(mots_normalises)

        # Étape 3 : Patterns regex
        if self.activer_regex:
            for pattern, remplacement in self._patterns_compiles:
                avant = resultat
                resultat = pattern.sub(remplacement, resultat)
                if resultat != avant:
                    modifications.append({
                        "type": "regex",
                        "pattern": pattern.pattern,
                        "apres": remplacement,
                    })

        return resultat, modifications

    def normaliser_page(
        self,
        lignes: list[dict],
        seuil_confiance_review: float = 0.6,
    ) -> list[LigneNormalisee]:
        """Normalise toutes les lignes d'une page.

        Args:
            lignes: Liste de dicts issus du data contract HTR.
                    Chaque dict doit avoir "id_ligne", "transcription", "confiance".
            seuil_confiance_review: Lignes sous ce seuil sont marquées needs_review.

        Returns:
            Liste de LigneNormalisee dans le même ordre que l'entrée.

        Example:
            >>> resultats = norm.normaliser_page(page["lignes"])
            >>> sum(1 for r in resultats if r.a_ete_modifie)
            12  # 12 lignes ont été modifiées
        """
        resultats = []

        for ligne in lignes:
            texte_original = ligne.get("transcription", "")
            confiance = ligne.get("confiance", 1.0)

            texte_normalise, modifications = self.normaliser_ligne(texte_original)

            resultats.append(LigneNormalisee(
                id_ligne=ligne.get("id_ligne", ""),
                texte_original=texte_original,
                texte_normalise=texte_normalise,
                modifications=modifications,
                confiance_htr=confiance,
                needs_review=confiance < seuil_confiance_review,
            ))

        return resultats


# ── Évaluation de la normalisation ────────────────────────────────────────────


def evaluer_normalisation(
    paires_avant: list[tuple[str, str]],
    paires_apres: list[tuple[str, str]],
) -> dict:
    """Compare le CER avant et après normalisation.

    Args:
        paires_avant: Liste (reference, hypothese_brute) avant normalisation.
        paires_apres: Liste (reference, hypothese_normalisee) après normalisation.

    Returns:
        Dictionnaire avec CER avant, CER après, amélioration absolue et relative.

    Example:
        >>> stats = evaluer_normalisation(paires_avant, paires_apres)
        >>> print(f"CER avant: {stats['cer_avant']:.2%} → après: {stats['cer_apres']:.2%}")
        CER avant: 18.50% → après: 12.30%
    """
    import editdistance

    def cer_global(paires):
        total_err = sum(editdistance.eval(r, h) for r, h in paires if r)
        total_chars = sum(len(r) for r, h in paires if r)
        return total_err / total_chars if total_chars > 0 else 0.0

    cer_avant = cer_global(paires_avant)
    cer_apres = cer_global(paires_apres)
    amelioration_abs = cer_avant - cer_apres
    amelioration_rel = amelioration_abs / cer_avant if cer_avant > 0 else 0.0

    return {
        "cer_avant": round(cer_avant, 4),
        "cer_apres": round(cer_apres, 4),
        "amelioration_absolue": round(amelioration_abs, 4),
        "amelioration_relative": round(amelioration_rel, 4),
        "nb_lignes": len(paires_avant),
    }


def statistiques_normalisation(
    lignes_normalisees: list[LigneNormalisee],
) -> dict:
    """Calcule des statistiques sur les modifications appliquées.

    Utile pour le README et les slides : montre l'impact chiffré des règles.

    Args:
        lignes_normalisees: Sortie de NormaliseurRegles.normaliser_page().

    Returns:
        Dictionnaire de statistiques détaillées.

    Example:
        >>> stats = statistiques_normalisation(resultats)
        >>> stats["taux_modification"]
        0.34  # 34% des lignes ont été modifiées
    """
    nb_total = len(lignes_normalisees)
    nb_modifiees = sum(1 for l in lignes_normalisees if l.a_ete_modifie)
    nb_needs_review = sum(1 for l in lignes_normalisees if l.needs_review)

    # Compter les types de modifications
    compteur_types: dict[str, int] = {}
    compteur_abrev: dict[str, int] = {}

    for ligne in lignes_normalisees:
        for mod in ligne.modifications:
            type_mod = mod.get("type", "inconnu")
            compteur_types[type_mod] = compteur_types.get(type_mod, 0) + 1

            if type_mod == "abreviation":
                avant = mod.get("avant", "")
                compteur_abrev[avant] = compteur_abrev.get(avant, 0) + 1

    # Top 10 abréviations les plus fréquentes
    top_abreviations = sorted(
        compteur_abrev.items(), key=lambda x: x[1], reverse=True
    )[:10]

    return {
        "nb_total_lignes": nb_total,
        "nb_lignes_modifiees": nb_modifiees,
        "taux_modification": round(nb_modifiees / nb_total, 4) if nb_total > 0 else 0,
        "nb_needs_review": nb_needs_review,
        "taux_needs_review": round(nb_needs_review / nb_total, 4) if nb_total > 0 else 0,
        "modifications_par_type": compteur_types,
        "top_10_abreviations": top_abreviations,
        "nb_total_modifications": sum(len(l.modifications) for l in lignes_normalisees),
    }


# ── Export ─────────────────────────────────────────────────────────────────────


def exporter_json_normalise(
    lignes_normalisees: list[LigneNormalisee],
    chemin_sortie: str | Path,
    statistiques: dict | None = None,
) -> None:
    """Exporte les transcriptions normalisées en JSON pour le module NLP.

    Args:
        lignes_normalisees: Résultats de la normalisation.
        chemin_sortie: Chemin du fichier JSON de sortie.
        statistiques: Statistiques optionnelles à inclure dans les métadonnées.

    Example:
        >>> exporter_json_normalise(resultats, "dataset_nlp/normalise.json", stats)
    """
    chemin = Path(chemin_sortie)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    sortie = {
        "metadata": {
            "version": "1.0.0",
            "etape": "normalisation",
            "statistiques": statistiques or {},
        },
        "lignes": [l.to_dict() for l in lignes_normalisees],
    }

    chemin.write_text(
        json.dumps(sortie, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"✅ {len(lignes_normalisees)} lignes exportées → {chemin}")
