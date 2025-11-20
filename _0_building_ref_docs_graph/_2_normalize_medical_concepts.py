import os
import json
import spacy
from scispacy.linking import EntityLinker
from typing import Dict, Any


# ============================================================
# 1. LOAD SCISPACY + UMLS LINKER
# ============================================================
print("🔬 Loading SciSpaCy model en_core_sci_sm...")
nlp = spacy.load("en_core_sci_sm")

print("🔗 Loading UMLS EntityLinker...")
nlp.add_pipe("scispacy_linker", config={
    "resolve_abbreviations": True,
    "linker_name": "umls"
})

linker = nlp.get_pipe("scispacy_linker")
kb = linker.kb


# ============================================================
# 2. NORMALIZE ANY TEXT INTO UMLS ENTITY
# ============================================================
def umls_normalize(text: str) -> Dict[str, Any]:

    original = text.strip()
    doc = nlp(original)

    # default fallback if nothing found
    default = {
        "text": original,
        "canonical_name": original.lower(),
        "cui": None,
        "score": None,
        "definition": None,
        "aliases": []
    }

    if not doc.ents:
        return default

    ent = doc.ents[0]
    candidates = ent._.kb_ents
    if not candidates:
        return default

    # take top candidate
    cui, score = candidates[0]
    info = kb.cui_to_entity[cui]

    return {
        "text": original,
        "canonical_name": info.canonical_name.lower(),
        "cui": cui,
        "score": float(score),
        "definition": info.definition,
        "aliases": info.aliases[:10]
    }


# ============================================================
# 3. NORMALIZE A CLAIM TRIPLET
# (adapted from KG version but for claim structure)
# ============================================================
def normalize_claim_fact(fact: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "subject": umls_normalize(fact["subject"]),
        "predicate": fact["predicate"],        # keep as is
        "object": umls_normalize(fact["object"]),
    }


# ============================================================
# 4. PROCESS A CLAIM JSONL FILE
# ============================================================
def normalize_claim_file(input_path: str, output_path: str):

    print(f"\n📄 Processing claim file: {os.path.basename(input_path)}")

    normalized_lines = []

    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            entry = json.loads(line)

            claim_id = entry.get("claim_id", "")
            claim_text = entry.get("claim", "")
            facts = entry.get("facts", [])

            # normalize each triple
            normalized_facts = [normalize_claim_fact(f) for f in facts]

            normalized_lines.append({
                "claim_id": claim_id,
                "claim": claim_text,
                "facts": normalized_facts
            })

    # write output
    with open(output_path, "w", encoding="utf-8") as outfile:
        for obj in normalized_lines:
            outfile.write(json.dumps(obj) + "\n")

    print(f"✅ Saved normalized → {output_path}")


# ============================================================
# 5. RECURSIVELY NORMALIZE ALL DATASET FOLDERS
# ============================================================
if __name__ == "__main__":

    input_root = "../data/claims/claim_extracted"
    output_root = "../data/claims/claim_extracted_umls"

    os.makedirs(output_root, exist_ok=True)

    print("\n=============== STARTING UMLS NORMALIZATION FOR CLAIMS ===============")

    for dataset_name in os.listdir(input_root):

        dataset_input_path = os.path.join(input_root, dataset_name)
        if not os.path.isdir(dataset_input_path):
            continue

        print(f"\n============== DATASET: {dataset_name} ==============")

        dataset_output_path = os.path.join(output_root, dataset_name)
        os.makedirs(dataset_output_path, exist_ok=True)

        # search for train.jsonl, dev.jsonl, test.jsonl
        for fname in os.listdir(dataset_input_path):

            if not fname.endswith(".jsonl"):
                continue

            input_file_path = os.path.join(dataset_input_path, fname)
            output_file_path = os.path.join(dataset_output_path, "umls_" + fname)

            normalize_claim_file(input_file_path, output_file_path)

    print("\n🎉 ALL CLAIM FILES UMLS-NORMALIZED SUCCESSFULLY!")
