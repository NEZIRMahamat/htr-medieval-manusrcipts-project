from __future__ import annotations
import json
import random
import os
from pathlib import Path
import numpy as np
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator, EarlyStoppingCallback
from src_htr.dataset import load_data_from_dir, extraire_exemples_htr, DatasetHTRMedieval
from src_htr.model import charger_trocr_lora

SEED = 42

def fixer_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ[chr(80)+chr(89)+chr(84)+chr(72)+chr(79)+chr(78)+chr(72)+chr(65)+chr(83)+chr(72)+chr(83)+chr(69)+chr(69)+chr(68)] = str(seed)

def entrainer(data_path=None, output_dir=None, lora_r=8, learning_rate=5e-5, nb_epochs=10, batch_size=8, max_train=5000, max_val=500, patience=3):
    data_path = data_path or str(Path(chr(100)+chr(97)+chr(116)+chr(97)) / (chr(109)+chr(101)+chr(100)+chr(105)+chr(101)+chr(118)+chr(97)+chr(108)+chr(95)+chr(100)+chr(97)+chr(116)+chr(97)))
    output_dir = output_dir or str(Path(chr(111)+chr(117)+chr(116)+chr(112)+chr(117)+chr(116)+chr(115)) / (chr(99)+chr(104)+chr(101)+chr(99)+chr(107)+chr(112)+chr(111)+chr(105)+chr(110)+chr(116)+chr(115)+chr(95)+chr(104)+chr(116)+chr(114)))
    fixer_seeds(SEED)
    print(chr(128194)+chr(32)+chr(67)+chr(104)+chr(97)+chr(114)+chr(103)+chr(101)+chr(109)+chr(101)+chr(110)+chr(116)+chr(32)+chr(100)+chr(117)+chr(32)+chr(100)+chr(97)+chr(116)+chr(97)+chr(115)+chr(101)+chr(116)+chr(32)+chr(58)+chr(32)+str(data_path))
    dataset = load_data_from_dir(data_path)
    exemples_train = extraire_exemples_htr(dataset, max_exemples=max_train)
    exemples_val = extraire_exemples_htr(dataset, split=chr(118)+chr(97)+chr(108)+chr(105)+chr(100)+chr(97)+chr(116)+chr(105)+chr(111)+chr(110), max_exemples=max_val)
    print(str(len(exemples_train)) + chr(32)+chr(124)+chr(32) + str(len(exemples_val)))
    modele, processeur = charger_trocr_lora(lora_r=lora_r)
    dataset_train = DatasetHTRMedieval(exemples_train, processeur)
    dataset_val = DatasetHTRMedieval(exemples_val, processeur)
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=nb_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy=chr(101)+chr(112)+chr(111)+chr(99)+chr(104),
        save_strategy=chr(101)+chr(112)+chr(111)+chr(99)+chr(104),
        load_best_model_at_end=True,
        predict_with_generate=True,
        seed=SEED,
        fp16=torch.cuda.is_available(),
    )
    trainer = Seq2SeqTrainer(
        model=modele,
        args=args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )
    print(chr(128640)+chr(32)+chr(68)+chr(233)+chr(109)+chr(97)+chr(114)+chr(114)+chr(97)+chr(103)+chr(101)+chr(32)+chr(102)+chr(105)+chr(110)+chr(101)+chr(45)+chr(116)+chr(117)+chr(110)+chr(105)+chr(110)+chr(103)+chr(46)+chr(46)+chr(46))
    trainer.train()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    modele.save_pretrained(output_dir)
    processeur.save_pretrained(output_dir)
    print(chr(9989)+chr(32)+chr(77)+chr(111)+chr(100)+chr(232)+chr(108)+chr(101)+chr(32)+chr(115)+chr(97)+chr(117)+chr(118)+chr(101)+chr(103)+chr(97)+chr(114)+chr(100)+chr(233)+chr(32)+chr(58)+chr(32)+str(output_dir))

if __name__ == chr(95)+chr(95)+chr(109)+chr(97)+chr(105)+chr(110)+chr(95)+chr(95):
    entrainer()