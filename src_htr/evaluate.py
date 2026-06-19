from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import editdistance
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import PeftModel
from src_htr.dataset import load_data_from_dir, extraire_exemples_htr
from src_htr.model import transcrire_ligne, MODELE_BASE

def calculer_cer(reference, hypothese):
    if not reference:
        return 0.0
    return editdistance.eval(reference, hypothese) / len(reference)

def evaluer_corpus(paires):
    total_err = sum(editdistance.eval(r, h) for r, h in paires if r)
    total_chars = sum(len(r) for r, h in paires if r)
    total_err_mots = sum(editdistance.eval(r.split(), h.split()) for r, h in paires if r)
    total_mots = sum(len(r.split()) for r, h in paires if r)
    return {
        chr(99)+chr(101)+chr(114)+chr(95)+chr(103)+chr(108)+chr(111)+chr(98)+chr(97)+chr(108): round(total_err / total_chars, 4) if total_chars else 0.0,
        chr(119)+chr(101)+chr(114)+chr(95)+chr(103)+chr(108)+chr(111)+chr(98)+chr(97)+chr(108): round(total_err_mots / total_mots, 4) if total_mots else 0.0,
        chr(110)+chr(98)+chr(95)+chr(108)+chr(105)+chr(103)+chr(110)+chr(101)+chr(115): len(paires),
    }

def bootstrap_cer(paires, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    cers = []
    for _ in range(n):
        idx = rng.integers(0, len(paires), size=len(paires))
        echantillon = [paires[i] for i in idx]
        metriques = evaluer_corpus(echantillon)
        cers.append(metriques[chr(99)+chr(101)+chr(114)+chr(95)+chr(103)+chr(108)+chr(111)+chr(98)+chr(97)+chr(108)])
    return {
        chr(99)+chr(101)+chr(114)+chr(95)+chr(109)+chr(101)+chr(100)+chr(105)+chr(97)+chr(110): round(float(np.median(cers)), 4),
        chr(105)+chr(99)+chr(95)+chr(98)+chr(97)+chr(115): round(float(np.percentile(cers, 2.5)), 4),
        chr(105)+chr(99)+chr(95)+chr(104)+chr(97)+chr(117)+chr(116): round(float(np.percentile(cers, 97.5)), 4),
    }

def main_evaluate(checkpoint_path=None, data_path=None, max_exemples=100):
    checkpoint_path = checkpoint_path or str(Path(chr(111)+chr(117)+chr(116)+chr(112)+chr(117)+chr(116)+chr(115)) / (chr(99)+chr(104)+chr(101)+chr(99)+chr(107)+chr(112)+chr(111)+chr(105)+chr(110)+chr(116)+chr(115)+chr(95)+chr(104)+chr(116)+chr(114)))
    data_path = data_path or str(Path(chr(100)+chr(97)+chr(116)+chr(97)) / (chr(109)+chr(101)+chr(100)+chr(105)+chr(101)+chr(118)+chr(97)+chr(108)+chr(95)+chr(100)+chr(97)+chr(116)+chr(97)))
    processeur = TrOCRProcessor.from_pretrained(MODELE_BASE)
    chemin = Path(checkpoint_path)
    if chemin.exists():
        base = VisionEncoderDecoderModel.from_pretrained(MODELE_BASE)
        modele = PeftModel.from_pretrained(base, str(chemin))
    else:
        modele = VisionEncoderDecoderModel.from_pretrained(MODELE_BASE)
    dataset = load_data_from_dir(data_path)
    exemples = extraire_exemples_htr(dataset, split=chr(116)+chr(101)+chr(115)+chr(116), max_exemples=max_exemples)
    resultats = []
    for i, ex in enumerate(exemples):
        texte, conf = transcrire_ligne(ex[chr(105)+chr(109)+chr(97)+chr(103)+chr(101)], modele, processeur)
        cer = calculer_cer(ex.get(chr(116)+chr(114)+chr(97)+chr(110)+chr(115)+chr(99)+chr(114)+chr(105)+chr(112)+chr(116)+chr(105)+chr(111)+chr(110), chr(32)), texte)
        resultats.append({chr(116)+chr(114)+chr(97)+chr(110)+chr(115)+chr(99)+chr(114)+chr(105)+chr(112)+chr(116)+chr(105)+chr(111)+chr(110): texte, chr(99)+chr(111)+chr(110)+chr(102)+chr(105)+chr(97)+chr(110)+chr(99)+chr(101): round(conf, 4), chr(99)+chr(101)+chr(114): round(cer, 4)})
        if (i+1) % 10 == 0:
            print(str(i+1) + chr(47) + str(len(exemples)))
    paires = [(ex.get(chr(116)+chr(114)+chr(97)+chr(110)+chr(115)+chr(99)+chr(114)+chr(105)+chr(112)+chr(116)+chr(105)+chr(111)+chr(110), chr(32)), r[chr(116)+chr(114)+chr(97)+chr(110)+chr(115)+chr(99)+chr(114)+chr(105)+chr(112)+chr(116)+chr(105)+chr(111)+chr(110)]) for ex, r in zip(exemples, resultats)]
    metriques = evaluer_corpus(paires)
    ic = bootstrap_cer(paires)
    print(chr(67)+chr(69)+chr(82)+chr(32)+chr(58)+chr(32)+str(metriques[chr(99)+chr(101)+chr(114)+chr(95)+chr(103)+chr(108)+chr(111)+chr(98)+chr(97)+chr(108)]))
    print(chr(87)+chr(69)+chr(82)+chr(32)+chr(58)+chr(32)+str(metriques[chr(119)+chr(101)+chr(114)+chr(95)+chr(103)+chr(108)+chr(111)+chr(98)+chr(97)+chr(108)]))
    print(chr(73)+chr(67)+chr(32)+chr(57)+chr(53)+chr(37)+chr(32)+chr(58)+chr(32)+str(ic[chr(105)+chr(99)+chr(95)+chr(98)+chr(97)+chr(115)])+chr(32)+str(ic[chr(105)+chr(99)+chr(95)+chr(104)+chr(97)+chr(117)+chr(116)]))
    Path(chr(111)+chr(117)+chr(116)+chr(112)+chr(117)+chr(116)+chr(115)+chr(47)+chr(109)+chr(101)+chr(116)+chr(114)+chr(105)+chr(99)+chr(115)).mkdir(parents=True, exist_ok=True)
    Path(chr(111)+chr(117)+chr(116)+chr(112)+chr(117)+chr(116)+chr(115)+chr(47)+chr(109)+chr(101)+chr(116)+chr(114)+chr(105)+chr(99)+chr(115)+chr(47)+chr(104)+chr(116)+chr(114)+chr(46)+chr(106)+chr(115)+chr(111)+chr(110)).write_text(json.dumps({chr(109)+chr(101)+chr(116)+chr(114)+chr(105)+chr(113)+chr(117)+chr(101)+chr(115): metriques, chr(98)+chr(111)+chr(111)+chr(116)+chr(115)+chr(116)+chr(114)+chr(97)+chr(112): ic}, indent=2), encoding=chr(117)+chr(116)+chr(102)+chr(45)+chr(56))

if __name__ == chr(95)+chr(95)+chr(109)+chr(97)+chr(105)+chr(110)+chr(95)+chr(95):
    main_evaluate()