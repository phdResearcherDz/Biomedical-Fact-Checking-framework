import os
import json
from pathlib import Path
import logging
from collections import defaultdict
from typing import Dict, List, Any

# ============================================================
# CONFIG
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("doc_aggregator")

RETRIEVAL_ROOT = Path("../data/retrievals/docs_kg_structure")
DECISION_MODEL_ROOT = Path("../data/decision_models")
DOC_DECISION_ROOT = Path("../data/document_level_decisions")

LABEL_SUPPORTED = "SUPPORTED"
LABEL_REFUTED = "REFUTED"
LABEL_NEI = "NOT ENOUGH INFO"

OUTPUT_LABEL_MAP = {
    LABEL_SUPPORTED: "Yes",
    LABEL_REFUTED: "No",
    LABEL_NEI: "NEI"
}

LABEL_TO_WEIGHT = {
    LABEL_SUPPORTED: +1,
    LABEL_REFUTED: -1,
    LABEL_NEI: 0,
}

# ============================================================
# 1. Fusion Logic (Eq. 17/18)
# ============================================================

def fuse_claims_for_document(claim_predictions, delta1=0.2, delta2=0.2):
    if not claim_predictions:
        return {"label": LABEL_NEI, "trust_score": 0.0}

    numerator = 0.0
    denom = 0.0

    for cp in claim_predictions:
        y_i = cp["predicted_label"]
        tau_i = float(cp["confidence"])

        if not y_i or tau_i <= 0:
            continue

        numerator += tau_i * LABEL_TO_WEIGHT.get(y_i, 0)
        denom += tau_i

    if denom == 0:
        tau_x = 0.0
        return {"label": LABEL_NEI, "trust_score": tau_x}

    tau_x = numerator / denom

    if tau_x > delta1:
        label = LABEL_SUPPORTED
    elif tau_x < -delta2:
        label = LABEL_REFUTED
    else:
        label = LABEL_NEI

    return {"label": label, "trust_score": float(tau_x)}

# ============================================================
# 2. Load retrieval + claim-level predictions
# ============================================================

def load_claim_to_evidence_map(dataset_name):
    path = RETRIEVAL_ROOT / dataset_name / "test.jsonl"
    if not path.exists():
        return {}

    mapping = {}
    with open(path, "r") as f:
        for line in f:
            r = json.loads(line)
            cid = r["claim_id"]
            mapping[cid] = {
                "claim_text": r["claim_text"],
                "evidence_doc_ids": r.get("evidence_doc_ids", []),
                "label": r.get("label", "UNKNOWN")
            }
    return mapping


def load_claim_level_predictions(dataset_name):
    path = DECISION_MODEL_ROOT / dataset_name / "claim_level_predictions.json"
    if not path.exists():
        return []

    return json.load(open(path, "r", encoding="utf-8"))

# ============================================================
# 3. Build document groups
# ============================================================

def build_document_key(evidence_doc_ids, fallback_id, dataset_name):
    if evidence_doc_ids:
        ids_str = ",".join(sorted(map(str, evidence_doc_ids)))
        return f"{dataset_name}::DOCS::{ids_str}"
    return f"{dataset_name}::NO_DOC::{fallback_id}"

# ============================================================
# 4. Aggregate for a dataset
# ============================================================

def aggregate_for_dataset(dataset_name, delta1=0.2, delta2=0.2):

    logger.info(f"\n=== Aggregating document decisions for dataset: {dataset_name} ===")

    claim_meta = load_claim_to_evidence_map(dataset_name)
    claim_preds = load_claim_level_predictions(dataset_name)

    docs = {}

    for cp in claim_preds:
        cid = cp["claim_id"]
        meta = claim_meta.get(cid, {})
        ev_ids = meta.get("evidence_doc_ids", [])
        claim_text = meta.get("claim_text", cp.get("claim_text", ""))

        dkey = build_document_key(ev_ids, cid, dataset_name)

        if dkey not in docs:
            docs[dkey] = {
                "dataset": dataset_name,
                "document_key": dkey,
                "evidence_doc_ids": ev_ids,
                "claims": []
            }

        docs[dkey]["claims"].append({
            "claim_id": cid,
            "claim_text": claim_text,
            "predicted_label": cp["predicted_label"],
            "confidence": float(cp["confidence"]),
            "gold_label": meta.get("label", "UNKNOWN")
        })

    logger.info(f"Built {len(docs)} document groups for {dataset_name}")

    # ====================================================
    # Save in requested format:
    # [
    #   {
    #     "id": "...",
    #     "text": "...",
    #     "predicated_label": "Yes/No/NEI",
    #     "trust_score": 0.012
    #   },
    #   ...
    # ]
    # ====================================================

    out_dir = DOC_DECISION_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "document_level_decisions.json"

    output_records = []

    for doc_key, d in docs.items():
        fusion = fuse_claims_for_document(d["claims"], delta1, delta2)

        final_label = fusion["label"]
        trust = fusion["trust_score"]

        text = d["claims"][0]["claim_text"] if d["claims"] else ""

        output_records.append({
            "id": doc_key,
            "text": text,
            "predicated_label": OUTPUT_LABEL_MAP[final_label],
            "trust_score": trust
        })

    json.dump(output_records, open(out_path, "w", encoding="utf-8"), indent=2)

    logger.info(f"Saved results → {out_path}")


# ============================================================
# MAIN
# ============================================================

def run_doc_aggregation(delta1=0.2, delta2=0.2):

    for dataset_dir in RETRIEVAL_ROOT.iterdir():
        if dataset_dir.is_dir():
            aggregate_for_dataset(dataset_dir.name, delta1, delta2)


if __name__ == "__main__":
    run_doc_aggregation()
