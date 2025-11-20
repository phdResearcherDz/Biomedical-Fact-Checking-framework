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

def normalize_pubmedqa_label(ans):
    """Map PubMedQA answers to fact-checker labels."""
    if ans is None:
        return "NOT ENOUGH INFO"

    ans = ans.strip().lower()

    if ans == "yes":
        return "SUPPORTED"
    if ans == "no":
        return "REFUTED"
    if ans == "maybe":
        return "NOT ENOUGH INFO"

    return "NOT ENOUGH INFO"


# ==========================
# Conversion logic
# ==========================

def convert_pubmedqa(dataset, tag_name):
    claims = []
    docs = []

    for item in dataset:
        cid = f"PMQA_{item['id']}"
        question = item["question"]
        context = item.get("context", "")
        answer = item.get("answer", None)

        label = normalize_pubmedqa_label(answer)

        claims.append({
            "claim_id": cid,
            "claim": question,
            "label": label if "test" not in tag_name else None,
            "evidence_doc_ids": [f"DOC_{cid}"] if "test" not in tag_name else [],
            "dataset": "PubMedQA"
        })

        docs.append({
            "id": f"DOC_{cid}",
            "title": question,
            "content": context
        })

    return claims, docs


# ==========================
# MAIN EXECUTION
# ==========================

RAW_DIR = Path("../data/claims/raw/PubMedQA")
OUT_DIR = Path("../data/claims/processed/PubMedQA")

os.makedirs(OUT_DIR / "claims", exist_ok=True)
os.makedirs(OUT_DIR / "docs", exist_ok=True)

# ===================================================================
# 1. PROCESS ORIGINAL SPLITS
# ===================================================================

original_files = {
    "train": RAW_DIR / "train.json",
    "dev":   RAW_DIR / "dev.json",
    "test":  RAW_DIR / "test.json"
}

for tag, path in original_files.items():
    if not path.exists():
        print(f"⚠ Missing: {path}, skipping.")
        continue

    print(f"Processing original split: {path}")

    data = load_json(path)
    claims, docs = convert_pubmedqa(data, tag)

    save_jsonl(claims, OUT_DIR / "claims" / f"{tag}.jsonl")
    save_jsonl(docs,   OUT_DIR / "docs"   / f"{tag}.jsonl")


# ===================================================================
# 2. PROCESS POISONED SUBFOLDERS
# ===================================================================

print("\n=== Processing poisoned PubMedQA splits ===")
poison_folders = [p for p in RAW_DIR.iterdir() if p.is_dir()]

for folder in poison_folders:
    print(f"\n--- Folder: {folder.name} ---")

    for f in folder.iterdir():
        if not f.name.endswith(".json"):
            continue

        file_tag = f"{folder.name}_{f.stem}"
        print(f"  • Processing {f.name} → tag = {file_tag}")

        dataset = load_json(f)
        claims, docs = convert_pubmedqa(dataset, file_tag)

        save_jsonl(claims, OUT_DIR / "claims" / f"{file_tag}.jsonl")
        save_jsonl(docs,   OUT_DIR / "docs"   / f"{file_tag}.jsonl")


# ===================================================================
# 3. BUILD UNIFIED CORPUS (all docs, clean + poisoned)
# ===================================================================

print("\n=== Building unified PubMedQA corpus ===")

corpus = []
docs_dir = OUT_DIR / "docs"

for f in docs_dir.iterdir():
    if f.suffix == ".jsonl" and f.name != "corpus.jsonl":
        with open(f, "r", encoding="utf-8") as reader:
            for line in reader:
                corpus.append(json.loads(line))

save_jsonl(corpus, docs_dir / "corpus.jsonl")

print("\n✅ PubMedQA clean + poisoned data successfully converted!")
