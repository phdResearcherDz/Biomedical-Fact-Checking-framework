import json
import os
from pathlib import Path

# ==========================
# Utility functions
# ==========================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_jsonl(list_obj, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for item in list_obj:
            f.write(json.dumps(item) + "\n")


# ==========================
# Label normalization
# ==========================

def normalize_bioasq_label(ans):
    if ans is None:
        return "NOT ENOUGH INFO"

    ans = ans.strip().lower()

    if ans == "yes":
        return "SUPPORTED"
    if ans == "no":
        return "REFUTED"
    return "NOT ENOUGH INFO"


# ==========================
# Single-file converter
# ==========================

def convert_bioasq(dataset, file_tag):
    """
    dataset → list of BioASQ-style entries
    file_tag → string used to identify split (train/dev/test/s1/etc.)
    """
    claims = []
    docs = []

    for item in dataset:
        cid = f"BAQ_{item['id']}"
        q = item["question"]
        context = item.get("context", "")
        answer = item.get("answer", None)

        label = normalize_bioasq_label(answer)

        claims.append({
            "claim_id": cid,
            "claim": q,
            "label": label if "test" not in file_tag else None,
            "evidence_doc_ids": [f"DOC_{cid}"] if "test" not in file_tag else [],
            "dataset": "BioASQ"
        })

        docs.append({
            "id": f"DOC_{cid}",
            "title": q,
            "content": context
        })

    return claims, docs


# ==========================
# MAIN EXECUTION
# ==========================

RAW_DIR = Path("../data/claims/raw/BioASQ")
OUT_DIR = Path("../data/claims/processed/BioASQ")

os.makedirs(OUT_DIR / "claims", exist_ok=True)
os.makedirs(OUT_DIR / "docs", exist_ok=True)

# ===================================================================
# 1. PROCESS ORIGINAL SPLITS (train.json, dev.json, test.json)
# ===================================================================

original_files = {
    "train": RAW_DIR / "train.json",
    "dev":   RAW_DIR / "dev.json",
    "test":  RAW_DIR / "test.json",
}

for tag, path in original_files.items():
    if not path.exists():
        print(f"⚠ Missing {path}, skipping.")
        continue

    print(f"Processing original split: {path}")

    data = load_json(path)
    claims, docs = convert_bioasq(data, tag)

    save_jsonl(claims, OUT_DIR / "claims" / f"{tag}.jsonl")
    save_jsonl(docs,   OUT_DIR / "docs"   / f"{tag}.jsonl")


# ===================================================================
# 2. PROCESS POISONED SUBFOLDERS
# ===================================================================

print("\n=== Processing poisoned BioASQ splits ===")
poison_folders = [p for p in RAW_DIR.iterdir() if p.is_dir()]

for folder in poison_folders:
    print(f"\n--- Folder: {folder.name} ---")

    for f in folder.iterdir():
        if not f.name.endswith(".json"):
            continue

        file_tag = f"{folder.name}_{f.stem}"
        print(f"  • Processing {f.name}  → tag={file_tag}")

        dataset = load_json(f)
        claims, docs = convert_bioasq(dataset, file_tag)

        save_jsonl(claims, OUT_DIR / "claims" / f"{file_tag}.jsonl")
        save_jsonl(docs,   OUT_DIR / "docs"   / f"{file_tag}.jsonl")


# ===================================================================
# 3. Build unified corpus
# ===================================================================

print("\n=== Building unified corpus ===")

corpus = []

for f in (OUT_DIR / "docs").iterdir():
    if f.name.endswith(".jsonl") and f.name != "corpus.jsonl":
        docs = []
        with open(f, "r", encoding="utf-8") as reader:
            for line in reader:
                docs.append(json.loads(line))
        corpus.extend(docs)

save_jsonl(corpus, OUT_DIR / "docs" / "corpus.jsonl")

print("\n✅ BioASQ + Poisoned splits successfully converted!")
