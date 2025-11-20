# Graph‑RAG Biomedical Fact‑Checking Framework

A full biomedical fact‑checking pipeline designed to **validate extracted SPO claims** from biomedical text.  
The system integrates **UMLS normalization**, **Graph‑RAG evidence retrieval**, **decision models**, and **vlaim and ocument‑level trust scoring**.

---

## 🚀 Features

### ✔ Claim Extraction

Extract SPO triplets using an LLM agent (Gemma‑3‑12B‑IT).

### ✔ UMLS Entity Normalization

Normalize biomedical entities using **SciSpaCy + UMLS linker**.

### ✔ Evidence Retrieval

Query dense FAISS indexes built from verified biomedical knowledge graphs.

### ✔ Claim‑Level Decision Model

Fine‑tune **PubMedBERT** to classify each claim into:

- **SUPPORTED**
- **REFUTED**
- **NOT ENOUGH INFO**

Provides **confidence scores** per claim.

### ✔ Document‑Level Trust Aggregation

Weighted fusion based on claim predictions → Global trust score τₓ.

### ✔ Dataset Formatters Included

We build formaters for each of these datasets:

- **SciFact**
- **BioASQ**
- **PubMedQA**
- **Poisoned splits (claim deletion, negation, hallucination…) from BioASQ and PubMedQA**
  and additionally:
- **CheckThat**
- **CoVERT**
- **Covid-Fact**
- **Fever**
- **HealthVer**
- **PUBHEALTH**
- **VitaminC**

---

## 📂 Project Structure

```
/
│
├── _0_building_ref_docs_graph/                # Step 0 — Build reference KG from trusted biomedical sources
│   │
│   ├── _1_extract_medical_triplets.py         # Extract SPO triplets from reference documents (LLM-based)
│   ├── _2_normalize_medical_concepts.py       # UMLS normalization using SciSpaCy
│   ├── _3_create_embedding_index.py           # Build FAISS vector index (ModernBERT embeddings)
│   ├── check_the_skipped_one.py               # Utility to re-process skipped documents
│   └── ...
│
├── _1_Claims_Extraction/                       # Step 1 — Extract and normalize claims from datasets
│   │
│   ├── _01_datasets_claim_extraction.py       # Extract claims via Gemma / LLM pipeline
│   └── _02_datasets_claim_normalization.py    # Normalize extracted claims (UMLS)
│
├── _2_Evidence_Extraction/                     # Step 2 — Retrieve KG evidence for each claim
│   │
│   ├── _1_evidence_extraction.py              # Builds embedding queries from normalized triplets
│   ├── _2_eval_retrival.py                    # Retrieval evaluation (Precision@K / Recall@K / MRR)
│   └── ...
│
├── _3_Decision_Making/                         # Step 3 — Claim-level & document-level decision making
│   │
│   ├── _1_train_decision_model_claim_level.py  # Train PubMedBERT-based classifier on retrieved evidence
│   ├── _2_final_decision.py                    # Weighted fusion → document-level factuality scores
│   └── utils/                                  # Helper functions for decision pipeline
│
├── data/
│   │
│   ├── claims/                                 # Processed claims (train/dev/test) for each dataset
│   ├── retrievals/                             # Evidence retrieved from KG for each claim
│   ├── decision_training_data/                 # Claim-level training dataset used by classifier
│   ├── document_level_decisions/               # Final Yes/No/NEI document-level outputs
│   └── ...
│
├── data_preprocessing/                         # Converters from raw dataset → unified fact-checking format
│   │
│   ├── _preprocess_bioasq.py                  # BioASQ Yes/No → Fact-checking JSONL conversion
│   ├── _preprocess_pubmedqa.py                # PubMedQA Yes/Maybe/No → Fact-checking JSONL
│   ├── _preprocess_scifact.py                 # SciFact → Fact-checking JSONL
│   ├── _preprocess_covid_fact.py              # Covid-Fact dataset conversion
│   ├── _preprocess_covert.py                  # CoVERT dataset conversion
│   ├── _preprocess_healthver.py               # HealthVer dataset conversion
│   ├── _preprocess_fever.py                   # FEVER dataset conversion
│   ├── _preprocess_pubrealth.py               # PUBHEALTH dataset conversion
│   ├── _preprocess_vitaminc.py                # VitaminC dataset conversion
│   └── poisoning variants/                    # s1.json, v8.json, etc., for poisoned versions
│
└── README.md                                   # Full documentation of the pipeline and usage

```

---

## 🛠 Requirements

### 🧬 Python

```
Python 3.10–3.11 recommended
```

### 📦 Python Packages

Install all dependencies:

```bash
pip install -r requirements.txt
```

Key packages:

- **transformers**
- **datasets**
- **torch**
- **scispacy**
- **spacy**
- **faiss-cpu / faiss-gpu**
- **langchain-community**
- **langchain-huggingface**

---

## 🖥 Hardware Used

Training and evaluation were performed on:

| Component | Specification                           |
| --------- | --------------------------------------- |
| **GPUs**  | 2× NVIDIA RTX 4060 Ti (16 GB VRAM each) |
| **RAM**   | 64 GB DDR5 @ 6000 MHz                   |
| **CPU**   | High‑core‑count AMD/Intel supported     |

### Performance Notes

- UMLS normalization is CPU‑heavy.
- FAISS search benefits significantly from GPU acceleration.
- Claim‑level classifier fine‑tuning runs efficiently on dual 4060 Ti GPUs.

---

## ⚙️ Pipeline Usage

### ### 1️⃣ Extract & Normalize Claims

```bash
python _0_building_ref_docs_graph/2_normalize_medical_concepts.py
```

### 2️⃣ Build Graph‑RAG Evidence Retrieval

```bash
python _1_evidence_retrieval/retrieval_pipeline.py
```

Produces:

```
data/retrievals/docs_kg_structure/<dataset>/test.jsonl
```

### 3️⃣ Train Claim‑Level Classifier

```bash
python _3_Decision Making/__train_decision_model.py
```

Outputs:

```
decision_models/<dataset>/claim_level_predictions.json
```

### 4️⃣ Document‑Level Aggregation

```bash
python _3_Decision Making/_2_final_decision.py
```

Outputs (formatted):

```
document_level_decisions/<dataset>/document_level_decisions.json
```

Example output:

```json
{
  "id": "513cdc38bee46bd34c000007",
  "text": "Is thrombophilia related to increased risk of miscarriage?",
  "predicated_label": "Yes",
  "trust_score": 0.0113
}
```

---

## 📘 Dataset Conversion Tools

Scripts included to convert raw datasets into the Fact‑Checking Framework format:

- **SciFact converter**
- **BioASQ Yes/No converter + poisoned splits**
- **PubMedQA converter + poisoned splits**
- **...**

Run them inside:

```
data/claims/raw/<dataset>/
```
