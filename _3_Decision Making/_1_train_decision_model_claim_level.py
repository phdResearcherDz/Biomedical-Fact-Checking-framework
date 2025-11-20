import os
import json
from pathlib import Path
import logging

import torch
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import DataCollatorWithPadding



# ============================================================
# CONFIG
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("decision_trainer")

RETRIEVAL_ROOT = Path("../data/retrievals/docs_kg_structure")
TRAINING_DATA_ROOT = Path("../data/decision_training_data")
OUTPUT_ROOT = Path("../data/decision_models")

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

LABELS = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]
label2id = {lbl: i for i, lbl in enumerate(LABELS)}
id2label = {i: lbl for lbl, i in label2id.items()}


# ============================================================
# Helpers
# ============================================================

def normalize_label(raw):
    """Ensure consistent final label space."""
    if raw is None:
        return "NOT ENOUGH INFO"

    r = raw.strip().upper()

    if r in ["SUPPORTS", "SUPPORTED"]:
        return "SUPPORTED"
    if r in ["REFUTES", "REFUTED"]:
        return "REFUTED"
    return "NOT ENOUGH INFO"


def aggregate_evidence(record):
    """Combine textual evidence found in retrieval."""
    ev_texts = []

    # Evidence retrieved per fact
    fact_bundles = record.get("retrieved_kg_relations_per_fact", [])
    for fb in fact_bundles:
        for rel in fb.get("retrieved_relations", []):
            if rel["page_content"]:
                ev_texts.append(rel["page_content"])

    # Evidence retrieved from full claim text
    ct = record.get("retrieved_documents_for_claim_text")
    if isinstance(ct, list):
        for item in ct:
            if item["page_content"]:
                ev_texts.append(item["page_content"])

    if not ev_texts:
        return "NO_EVIDENCE_FOUND"

    return " ".join(ev_texts)


def build_training_data(dataset_name):
    """
    Convert retrieval output → training dataset for PubMedBERT classifier.
    """
    dataset_dir = RETRIEVAL_ROOT / dataset_name
    output_dir = TRAINING_DATA_ROOT / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Building training data for {dataset_name} ===")

    for split in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
        in_path = dataset_dir / split
        out_path = output_dir / split.replace(".jsonl", ".json")

        if not in_path.exists():
            logger.warning(f"Missing file: {in_path}")
            continue

        logger.info(f"Processing {in_path} → {out_path}")

        output_records = []
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)

                out = {
                    "claim_id": r["claim_id"],
                    "claim_text": r["claim_text"],
                    "evidence": aggregate_evidence(r),
                    "label": normalize_label(r.get("label")),
                }
                output_records.append(out)

        with open(out_path, "w", encoding="utf-8") as f:
            for o in output_records:
                f.write(json.dumps(o) + "\n")

    logger.info("✔ Training data generated.\n")


# ============================================================
# Training helpers
# ============================================================

def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def load_training_dataset(dataset_name):
    """Load processed JSON training data."""
    root = TRAINING_DATA_ROOT / dataset_name

    data_files = {}
    for split in ["train", "dev", "test"]:
        f = root / f"{split}.json"
        if f.exists():
            data_files[split] = str(f)

    if not data_files:
        raise ValueError(f"No training data found for {dataset_name}")

    ds = load_dataset("json", data_files=data_files)

    # Convert raw label → numeric
    def add_labels(example):
        lbl = normalize_label(example["label"])
        example["labels"] = label2id[lbl]
        return example

    ds = ds.map(add_labels)

    # ❗ Remove string label column so Trainer does NOT try to tensorize it
    if "label" in ds["train"].column_names:
        ds = ds.remove_columns(["label"])

    return ds


# ============================================================
# Train PubMedBERT decision classifier
# ============================================================

def train_decision_model(dataset_name):
    logger.info(f"=== Training decision model for {dataset_name} ===")

    ds = load_training_dataset(dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def preprocess(examples):
        return tokenizer(
            examples["claim_text"],
            examples["evidence"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    tokenized = ds.map(preprocess, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    output_dir = OUTPUT_ROOT / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("dev", tokenized.get("test")),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("=== Training classifier ===")
    trainer.train()

    # ============================
    # 1) Evaluate final model
    # ============================
    test_ds = tokenized.get("test", tokenized.get("dev"))
    eval_metrics = trainer.evaluate(eval_dataset=test_ds)

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=4)

    logger.info(f"✔ Test metrics saved to {output_dir / 'test_metrics.json'}")

    # ============================
    # 2) CLAIM-LEVEL PREDICTIONS
    # ============================
    logger.info("=== Generating claim-level predictions with confidence ===")

    # Run prediction
    predictions = trainer.predict(test_ds)

    logits = predictions.predictions
    softmax = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_ids = softmax.argmax(axis=1)

    # Original raw test data (not tokenized)
    raw_test = ds.get("test", ds.get("dev"))

    claim_level_output = []
    for i, example in enumerate(raw_test):
        claim_level_output.append({
            "claim_id": example["claim_id"],
            "claim_text": example["claim_text"],
            "predicted_label": id2label[int(pred_ids[i])],
            "confidence": float(softmax[i][pred_ids[i]])
        })

    # Save predictions
    pred_path = output_dir / "claim_level_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(claim_level_output, f, indent=4)

    logger.info(f"✔ Claim-level predictions saved to: {pred_path}")


# ============================================================
# MAIN
# ============================================================

def run_pipeline():
    logger.info("=== Starting decision-making training pipeline ===")

    for dataset_dir in RETRIEVAL_ROOT.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name

        logger.info(f"\n==============================")
        logger.info(f"DATASET: {dataset_name}")
        logger.info("==============================\n")

        # 1) Build training data
        build_training_data(dataset_name)

        # 2) Train classifier
        train_decision_model(dataset_name)


if __name__ == "__main__":
    run_pipeline()
