import os
import json
import re
import time
from typing import List, Dict, Set
from pydantic import BaseModel, ValidationError, constr

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama.llms import OllamaLLM


# ============================================================
# 1. Pydantic Schema (identical to reference-doc pipeline)
# ============================================================
class MedicalFact(BaseModel):
    subject: constr(strip_whitespace=True, min_length=1)
    predicate: constr(strip_whitespace=True, min_length=1)
    object: constr(strip_whitespace=True, min_length=1)


class FactExtractionOutput(BaseModel):
    facts: List[MedicalFact]


# Only extract first N CLAIMS per split for testing
MAX_CLAIMS_PER_FILE = 3   # set to None for full dataset extraction


# ============================================================
# 2. LLM + PROMPT (identical to reference-doc pipeline)
# ============================================================
llm = OllamaLLM(model="gemma3:12b")  # You already use this


extraction_prompt_template = PromptTemplate(
    input_variables=["sentence", "full_text", "previous_facts", "format_instructions"],
    template="""
You are a biomedical information extraction agent. Your task is to extract semantic, medically meaningful,
and non-redundant scientific claims from a sentence.

You MUST output a list of Subject–Predicate–Object (SPO) triplets.

-----------------------
### GLOBAL RULES
-----------------------
1. Extract ONLY biomedical factual assertions.
2. Use the full_text as global context to resolve pronouns, coreference, abbreviations.
3. Subjects and objects must be biomedical entities, NOT pronouns.
4. Predicates must be strong biomedical relations such as:
   - treats / prevents / causes / increases risk / decreases risk
   - induces / inhibits / reduces / associates with
   - expresses / regulates / activates / suppresses
5. Do NOT include weak relations (“helps”, “may help”, etc.)
6. Do NOT output duplicates found in previous_facts.
7. Normalize verb phrases (“helps reduce X” → “reduces X”).

-----------------------
### FEW-SHOT EXAMPLES
-----------------------
Sentence: "Ibuprofen helps reduce inflammation."
Output: [
  {{ "subject": "Ibuprofen", "predicate": "reduces", "object": "inflammation" }}
]

Sentence: "Metformin is used to treat type 2 diabetes."
Output: [
  {{ "subject": "Metformin", "predicate": "treats", "object": "type 2 diabetes" }}
]

Sentence: "ACE inhibitors lower blood pressure."
Output: [
  {{ "subject": "ACE inhibitors", "predicate": "reduce", "object": "blood pressure" }}
]

-----------------------
### SENTENCE TO PROCESS
{sentence}

### FULL DOCUMENT CONTEXT
{full_text}

### PREVIOUSLY EXTRACTED FACTS
{previous_facts}

Return output strictly in JSON format:
{format_instructions}
"""
)



extraction_parser = JsonOutputParser(pydantic_object=FactExtractionOutput)
extraction_chain = extraction_prompt_template | llm | extraction_parser


# ============================================================
# 3. Sentence & fact normalization helpers
# ============================================================
def split_into_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def normalize_fact_key(fact: Dict[str, str]) -> str:
    """Unique signature for deduplication."""
    return " | ".join([
        normalize_text(fact["subject"]),
        normalize_text(fact["predicate"]),
        normalize_text(fact["object"])
    ])


# ============================================================
# 4. Core LLM extraction functions (same as ref-doc version)
# ============================================================
def extract_facts_from_sentence(sentence, full_text, previous_facts, retries=5):
    attempt = 0
    fmt = extraction_parser.get_format_instructions()

    while attempt < retries:
        try:
            result = extraction_chain.invoke({
                "sentence": sentence,
                "full_text": full_text,
                "previous_facts": json.dumps(previous_facts, indent=2),
                "format_instructions": fmt
            })

            extracted = result.get("facts", [])

            valid = []
            for f in extracted:
                try:
                    mf = MedicalFact(**f)
                    valid.append(mf.model_dump())
                except:
                    continue

            return [{"fact": v, "sentence": sentence} for v in valid]

        except Exception as e:
            print(f"⚠️ Retry {attempt+1}: {e}")
            attempt += 1
            time.sleep(0.3)

    print(f"❌ Failed for sentence: {sentence[:80]}...")
    return []


def extract_from_text(text: str) -> Dict:
    sentences = split_into_sentences(text)
    all_facts = []
    seen = set()

    for i, s in enumerate(sentences):
        print(f"   🔍 Sentence {i+1}/{len(sentences)}: {s[:100]}...")
        prev = [x["fact"] for x in all_facts]

        new = extract_facts_from_sentence(s, text, prev)

        for item in new:
            f = item["fact"]
            key = normalize_fact_key(f)

            if key not in seen:
                seen.add(key)
                all_facts.append(item)

    return {"facts_with_sentence": all_facts}


# ============================================================
# 5. APPLY EXTRACTION TO CLAIM DATASETS
# ============================================================
def clean_claim_text(text: str) -> str:
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


input_root = "../data/claims/processed"
output_root = "../data/claims/claim_extracted"
os.makedirs(output_root, exist_ok=True)

skip_datasets = ["CoVERT", "Fever", "Covid-Fact", "HealthVer", "PubHealth", "VitaminC"]#"SciFact"

print("\n=============== START CLAIM TRIPLET EXTRACTION ===============")

for dataset_name in os.listdir(input_root):

    dataset_path = os.path.join(input_root, dataset_name)
    if not os.path.isdir(dataset_path) or dataset_name in skip_datasets:
        continue

    claims_path = os.path.join(dataset_path, "claims")
    if not os.path.exists(claims_path):
        print(f"⚠️ No claims/ folder in {dataset_name}")
        continue

    out_folder = os.path.join(output_root, dataset_name)
    os.makedirs(out_folder, exist_ok=True)

    for split in ["train.jsonl", "dev.jsonl", "test.jsonl"]:

        in_file = os.path.join(claims_path, split)
        if not os.path.exists(in_file):
            print(f"ℹ️ Missing split {split} in {dataset_name}")
            continue

        print(f"\n📘 Dataset: {dataset_name} — Split: {split}")

        enriched = []
        processed = 0

        with open(in_file, "r", encoding="utf-8") as f:
            for line in f:
                if MAX_CLAIMS_PER_FILE and processed >= MAX_CLAIMS_PER_FILE:
                    break

                obj = json.loads(line)
                claim = clean_claim_text(obj.get("claim", ""))
                claim_id = obj.get("claim_id", "")

                if not claim:
                    continue

                print(f"\n➡️ Processing claim: {claim[:120]}...")

                result = extract_from_text(claim)
                facts = [x["fact"] for x in result["facts_with_sentence"]]

                enriched.append({
                    "claim_id": claim_id,
                    "claim": claim,
                    "facts": facts
                })

                processed += 1

        out_file = os.path.join(out_folder, split)
        with open(out_file, "w", encoding="utf-8") as f:
            for row in enriched:
                f.write(json.dumps(row) + "\n")

        print(f"✅ Saved → {out_file}")

print("\n🎉 CLAIM EXTRACTION FINISHED SUCCESSFULLY!")
