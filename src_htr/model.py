from __future__ import annotations
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import get_peft_model, LoraConfig, TaskType

MODELE_BASE = chr(109)+chr(105)+chr(99)+chr(114)+chr(111)+chr(115)+chr(111)+chr(102)+chr(116)+chr(47)+chr(116)+chr(114)+chr(111)+chr(99)+chr(114)+chr(45)+chr(98)+chr(97)+chr(115)+chr(101)+chr(45)+chr(104)+chr(97)+chr(110)+chr(100)+chr(119)+chr(114)+chr(105)+chr(116)+chr(116)+chr(101)+chr(110)
MAX_LONGUEUR = 128

def charger_trocr_lora(lora_r=8, lora_alpha=32, lora_dropout=0.1):
    processeur = TrOCRProcessor.from_pretrained(MODELE_BASE)
    modele = VisionEncoderDecoderModel.from_pretrained(MODELE_BASE)
    modele.config.decoder_start_token_id = processeur.tokenizer.cls_token_id
    modele.config.pad_token_id = processeur.tokenizer.pad_token_id
    modele.config.vocab_size = modele.config.decoder.vocab_size
    modele.config.max_length = MAX_LONGUEUR
    modele.config.num_beams = 4
    config_lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[chr(113)+chr(95)+chr(112)+chr(114)+chr(111)+chr(106), chr(118)+chr(95)+chr(112)+chr(114)+chr(111)+chr(106)],
        bias=chr(110)+chr(111)+chr(110)+chr(101),
    )
    modele = get_peft_model(modele, config_lora)
    modele.print_trainable_parameters()
    return modele, processeur

def transcrire_ligne(image, modele, processeur, num_beams=4):
    from PIL import Image as PILImage
    import numpy as np
    if isinstance(image, np.ndarray):
        image = PILImage.fromarray(image).convert(chr(82)+chr(71)+chr(66))
    device = next(modele.parameters()).device
    pixel_values = processeur(images=image, return_tensors=chr(112)+chr(116)).pixel_values.to(device)
    with torch.no_grad():
        sorties = modele.generate(pixel_values, num_beams=num_beams, output_scores=True, return_dict_in_generate=True)
    texte = processeur.batch_decode(sorties.sequences, skip_special_tokens=True)[0]
    if hasattr(sorties, chr(115)+chr(99)+chr(111)+chr(114)+chr(101)+chr(115)) and sorties.scores:
        probs = [torch.softmax(s, dim=-1).max().item() for s in sorties.scores]
        confiance = float(np.mean(probs))
    else:
        confiance = 1.0
    return texte, confiance