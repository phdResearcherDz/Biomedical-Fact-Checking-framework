import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
import re


# =========================================================
# 1. CONFIG
# =========================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("retrieval_eval")

INPUT_ROOT = Path("../data/retrievals/docs_kg_structure")     # YOUR RETRIEVAL OUTPUT
OUTPUT_ROOT = Path("./evaluation_results")                    # WHERE TO SAVE RESULTS

K_VALUES = [1, 3, 5, 10, 20, 30, 40, 50]
MAX_RANK_FOR_HISTOGRAM = K_VALUES[-1]

LABEL_SUPPORTED = "SUPPORTED"
LABEL_REFUTED = "REFUTED"
LABEL_NEI = "NOT ENOUGH INFO"
LABEL_UNKNOWN = "UNKNOWN"

ORDERED_LABELS = [LABEL_SUPPORTED, LABEL_REFUTED, LABEL_NEI, LABEL_UNKNOWN]



# =========================================================
# 2. HELPERS
# =========================================================

def normalize_label(label):
    if not label:
        return LABEL_UNKNOWN
    label = label.strip().upper()
    if label in ["SUPPORTS", "SUPPORTED"]:
        return LABEL_SUPPORTED
    if label in ["REFUTES", "REFUTED"]:
        return LABEL_REFUTED
    if "NOT ENOUGH" in label or "NEI" in label:
        return LABEL_NEI
    return LABEL_UNKNOWN


def precision_at_k(retrieved, gt_set, k):
    if k == 0:
        return 0
    return sum(1 for d in retrieved[:k] if d in gt_set) / k


def recall_at_k(retrieved, gt_set, total_relevant):
    if total_relevant == 0:
        return 0
    return sum(1 for d in retrieved if d in gt_set) / total_relevant


def f1(p, r):
    if p + r == 0:
        return 0
    return 2 * p * r / (p + r)


def reciprocal_rank(retrieved, gt_set):
    for idx, doc_id in enumerate(retrieved):
        if doc_id in gt_set:
            return 1.0 / (idx + 1), idx + 1
    return 0.0, float("inf")



# =========================================================
# 3. MAIN EVALUATION
# =========================================================

def main():

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_output_path = OUTPUT_ROOT / "summary.json"
    detailed_output_path = OUTPUT_ROOT / "detailed_per_claim.jsonl"

    per_label_metrics = defaultdict(lambda: defaultdict(list))
    per_file_ranks = defaultdict(lambda: defaultdict(list))

    detailed_out = open(detailed_output_path, "w", encoding="utf-8")

    for dataset_folder in INPUT_ROOT.iterdir():
        if not dataset_folder.is_dir():
            continue

        dataset_name = dataset_folder.name
        logger.info(f"\nProcessing dataset: {dataset_name}")

        # auto-detect splits
        jsonl_files = list(dataset_folder.glob("*.jsonl"))
        if not jsonl_files:
            logger.warning(f"No JSONL outputs in {dataset_name}")
            continue

        for file_path in jsonl_files:

            split_name = file_path.stem
            file_id = f"{dataset_name}_{split_name}"

            logger.info(f" → Split: {split_name}")

            with open(file_path, "r", encoding="utf-8") as f:
                records = [json.loads(x) for x in f]

            for rec in records:

                claim_id = rec["claim_id"]
                label = normalize_label(rec.get("label"))
                gt_docs = set(rec.get("evidence_doc_ids", []))
                if not gt_docs:
                    continue

                # unify fact + claim-text retrieval
                retrieved_all = []

                # fact-based retrieval
                for fact_bundle in rec.get("retrieved_kg_relations_per_fact", []):
                    for rel in fact_bundle.get("retrieved_relations", []):
                        docid = rel["metadata"].get("document_id")
                        if docid:
                            retrieved_all.append((docid, rel["score"]))

                # claim-text retrieval
                if rec.get("retrieved_documents_for_claim_text"):
                    for rel in rec["retrieved_documents_for_claim_text"]:
                        docid = rel["metadata"].get("document_id")
                        if docid:
                            retrieved_all.append((docid, rel["score"]))

                if not retrieved_all:
                    # no retrieval → everything 0
                    per_label_metrics[label]["rr"].append(0.0)
                    for k in K_VALUES:
                        per_label_metrics[label][f"precision@{k}"].append(0.0)
                        per_label_metrics[label][f"recall@{k}"].append(0.0)
                        per_label_metrics[label][f"f1@{k}"].append(0.0)
                    continue

                # deduplicate (keep best score)
                dedup = {}
                for docid, score in retrieved_all:
                    if docid not in dedup or score < dedup[docid]:
                        dedup[docid] = score

                retrieved_sorted = [x[0] for x in sorted(dedup.items(), key=lambda z: z[1])]

                rr, first_rank = reciprocal_rank(retrieved_sorted, gt_docs)
                per_label_metrics[label]["rr"].append(rr)
                per_file_ranks[file_id][label].append(first_rank)

                # compute metrics @K
                for k in K_VALUES:
                    p = precision_at_k(retrieved_sorted, gt_docs, k)
                    r = recall_at_k(retrieved_sorted, gt_docs, len(gt_docs))
                    f1_k = f1(p, r)

                    per_label_metrics[label][f"precision@{k}"].append(p)
                    per_label_metrics[label][f"recall@{k}"].append(r)
                    per_label_metrics[label][f"f1@{k}"].append(f1_k)

                detailed_out.write(json.dumps({
                    "dataset": dataset_name,
                    "split": split_name,
                    "claim_id": claim_id,
                    "label": label,
                    "retrieved": retrieved_sorted,
                    "ground_truth": list(gt_docs),
                    "reciprocal_rank": rr,
                    "first_rank": None if first_rank == float("inf") else first_rank
                }) + "\n")

    detailed_out.close()

    # =========================================================
    # 4. SUMMARY METRICS
    # =========================================================

    final_summary = {}

    for label in ORDERED_LABELS:
        metrics = per_label_metrics[label]
        if not metrics:
            continue
        final_summary[label] = {}

        # RR
        final_summary[label]["MRR"] = float(np.mean(metrics["rr"])) if metrics["rr"] else 0

        # P/R/F1
        for k in K_VALUES:
            P = np.mean(metrics[f"precision@{k}"]) if metrics[f"precision@{k}"] else 0
            R = np.mean(metrics[f"recall@{k}"]) if metrics[f"recall@{k}"] else 0
            F = np.mean(metrics[f"f1@{k}"]) if metrics[f"f1@{k}"] else 0
            final_summary[label][f"precision@{k}"] = float(P)
            final_summary[label][f"recall@{k}"] = float(R)
            final_summary[label][f"f1@{k}"] = float(F)

    with open(summary_output_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=4)

    logger.info(f"\nSummary metrics saved to: {summary_output_path}")
    logger.info(f"Detailed metrics saved to: {detailed_output_path}")



if __name__ == "__main__":
    main()
