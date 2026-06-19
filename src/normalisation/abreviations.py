"""
Table des abréviations médiévales françaises et latines.

Sources :
- CREMMA Medieval conventions
- Cappelli, A. (1899). Dizionario di abbreviature latine ed italiane.
- Muzerelle, D. (1985). Vocabulaire codicologique.

Cette table couvre les abréviations les plus fréquentes dans les manuscrits
français du XIIIe au XVe siècle.
"""

# ── Abréviations par substitution directe ─────────────────────────────────────
# Format : "abréviation dans le texte HTR" → "expansion"

ABREVIATIONS_DIRECTES: dict[str, str] = {
    # Signes tironiens et symboles courants
    "&": "et",
    "⁊": "et",          # esperluette tironnienne
    "ꝑ": "per",         # p barré
    "ꝓ": "pro",         # p avec boucle
    "ꝗ": "que",         # q barré
    "ꝙ": "qui",
    "ꞃ": "rum",         # r rotunda + tilde

    # Abréviations fréquentes vieux/moyen français
    "qe": "que",
    "qi": "qui",
    "qnt": "quant",
    "cōme": "comme",
    "dōt": "dont",
    "sōt": "sont",
    "nre": "notre",
    "pre": "père",
    "mre": "mère",
    "sre": "sire",
    "dre": "dire",
    "fre": "frère",
    "vtre": "votre",
    "ltre": "lettre",
    "ēt": "est",
    "sñr": "seigneur",
    "sgr": "seigneur",
    "sr": "seigneur",
    "mr": "messire",
    "msr": "monseigneur",
    "mgr": "monseigneur",
    "st": "saint",
    "ste": "sainte",

    # Latin fréquent dans les manuscrits français
    "dñs": "dominus",
    "dns": "dominus",
    "ihs": "ihesus",
    "xps": "christus",
    "scs": "sanctus",
    "sca": "sancta",
    "scm": "sanctum",
    "eps": "episcopus",
    "pbr": "presbyter",
    "ff": "fratres",
    "pp": "papa",
    "nr": "noster",
    "vr": "vester",
    "om": "omnem",
    "qm": "quoniam",
    "qs": "quos",
    "qi": "qui",

    # Chiffres romains courants (normalisation optionnelle)
    # Commentés car on peut vouloir les garder
    # "ij": "2", "iij": "3", "iiij": "4",
}

# ── Abréviations par suffixe (tilde de nasale) ────────────────────────────────
# Le tilde ~ au-dessus d'une voyelle indique une nasale supprimée
# Format : suffixe abrégé → suffixe développé

NASALES_TILDE: dict[str, str] = {
    "ã": "an",   # a tilde → an
    "ẽ": "en",   # e tilde → en
    "ĩ": "in",   # i tilde → in
    "õ": "on",   # o tilde → on
    "ũ": "un",   # u tilde → un
    "ā": "an",   # a macron (variante)
    "ē": "en",   # e macron
    "ī": "in",   # i macron
    "ō": "on",   # o macron
    "ū": "un",   # u macron
}

# ── Abréviations par contexte (regex) ─────────────────────────────────────────
# Patterns plus complexes traités dans le module de normalisation

PATTERNS_REGEX: list[tuple[str, str]] = [
    # Tilde de nasale sur n'importe quelle voyelle (pattern Unicode)
    (r"([aeiouAEIOU])\u0303", r"\1n"),   # voyelle + combining tilde → voyelle + n
    (r"([aeiouAEIOU])\u0304", r"\1n"),   # voyelle + combining macron

    # p barré (per/par/por)
    (r"\bp̄\b", "par"),
    (r"\bp̃\b", "per"),

    # Superscripts courants
    (r"(\w+)\^r\b", r"\1re"),    # ex: m^r → mre → messire
    (r"(\w+)\^e\b", r"\1e"),
    (r"(\w+)\^s\b", r"\1s"),

    # Lacunes et zones illisibles (standardisation)
    (r"\[\.{3}\]", "[lacune]"),
    (r"\[†\]", "[endommagé]"),

    # Normalisation des u/v médiévaux (optionnel, à activer selon convention)
    # (r"\bv([aeiou])", r"u\1"),  # v initial devant voyelle → u (ex: "vn" → "un")
    # (r"\bi\b", "j"),             # i consonantique → j
]

# ── Corrections graphiques fréquentes ─────────────────────────────────────────
# Erreurs HTR typiques sur les manuscrits médiévaux

CORRECTIONS_HTR: dict[str, str] = {
    # Confusions visuelles fréquentes de TrOCR sur l'écriture gothique
    "rn": "m",      # 'rn' souvent confondu avec 'm' en gothique
    "ii": "u",      # double i souvent confondu avec u
    "vv": "w",
    "ij": "ij",     # à garder tel quel (graphie médiévale valide)
}
