import os
import json
import shutil
import numpy as np
import torch
import faiss
from tqdm import tqdm

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------- CONFIG -------------------

# 🧪 DEBUG mode: only index a few samples
TEST_MODE = True
TEST_LIMIT = 5

# UMLS text construction
USE_UMLS_CANONICAL = True
USE_UMLS_CUI = True
USE_RELATION = True
USE_FULL_TRIPLET = False  # if True overrides the others

# dataset folders (your normalized KG)
UMLS_KG_ROOT = "../data/kg_from_ref_docs_umls_normalized"

FAISS_INDEX_BASE_OUTPUT_DIR = "../data/Primary Evidence Source/umls_triplet_faiss_indexes"

MODEL_NAME = "lightonai/modernbert-embed-large"
EMBED_BATCH = 32


# ----------- HELPER: Prepare embedding text --------------

def build_embedding_text(tr):
    """
    Builds the text representation that will be embedded in ModernBERT.
    Options controlled by config flags.
    """

    subj = tr["subject"]
    obj = tr["object"]
    pred = tr["predicate"]

    # override using original format
    if USE_FULL_TRIPLET:
        return f"{subj['text']} [REL] {pred} [TAIL] {obj['text']}"

    parts = []

    # Subject
    if USE_UMLS_CANONICAL:
        parts.append(subj["canonical_name"])
    else:
        parts.append(subj["text"])

    if USE_UMLS_CUI and subj["cui"]:
        parts.append(f"(CUI:{subj['cui']})")

    # Relation
    if USE_RELATION:
        parts.append(pred)

    # Object
    if USE_UMLS_CANONICAL:
        parts.append(obj["canonical_name"])
    else:
        parts.append(obj["text"])

    if USE_UMLS_CUI and obj["cui"]:
        parts.append(f"(CUI:{obj['cui']})")

    return " ".join(parts)


# ----------- LOAD NORMALIZED KG FILES -------------------

def load_umls_triplets(dataset_folder):

    items = []

    for fname in os.listdir(dataset_folder):
        if not fname.startswith("umls_") or not fname.endswith(".json"):
            continue

        file_path = os.path.join(dataset_folder, fname)
        print(f"📄 Loading: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            triplets = json.load(f)

        for tr in triplets:
            embed_text = build_embedding_text(tr)

            items.append({
                "embed_text": embed_text,
                "page_content": tr["sentence"],
                "metadata": {
                    "document_id": tr["document_id"],
                    "context": tr["context"],
                    "sentence": tr["sentence"],
                    "full_content": tr["full_content"],
                    "umls_subject": tr["subject"],
                    "umls_object": tr["object"],
                    "relation": tr["predicate"],
                }
            })

    print(f"  → Loaded {len(items)} total triplets")
    return items


# ----------- Build FAISS Index ----------------------

def build_faiss_index(docs, embed_model):

    texts = [doc.metadata["embed_text_final"] for doc in docs]

    print(f"🔍 Embedding {len(texts)} items...")

    # embed in batches
    all_embeddings = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH), desc="Embedding"):
        batch = texts[i:i+EMBED_BATCH]
        vecs = embed_model.embed_documents(batch)
        all_embeddings.extend(vecs)

    emb_np = np.array(all_embeddings).astype("float32")

    print(f"📐 Embedding shape: {emb_np.shape}")

    faiss.normalize_L2(emb_np)

    index = faiss.IndexFlatL2(emb_np.shape[1])
    index.add(emb_np)

    print(f"📦 FAISS index size: {index.ntotal}")

    doc_ids = [str(i) for i in range(len(docs))]
    docstore = InMemoryDocstore({doc_id: doc for doc_id, doc in zip(doc_ids, docs)})
    id_map = {i: doc_id for i, doc_id in enumerate(doc_ids)}

    return FAISS(
        embedding_function=embed_model.embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=id_map,
        normalize_L2=False,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )


# ---------------- MAIN --------------------------

print("🚀 Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": 0 if torch.cuda.is_available() else -1},
    encode_kwargs={"normalize_embeddings": True},
)

for dataset_name in os.listdir(UMLS_KG_ROOT):

    dataset_path = os.path.join(UMLS_KG_ROOT, dataset_name)
    if not os.path.isdir(dataset_path):
        continue

    print(f"\n==================== Dataset: {dataset_name} ====================")

    # load data
    items = load_umls_triplets(dataset_path)

    if TEST_MODE:
        print(f"🧪 TEST MODE ACTIVE → Using first {TEST_LIMIT} samples")
        items = items[:TEST_LIMIT]

    # build Documents
    docs = []
    for it in items:
        meta = it["metadata"]
        meta["embed_text_final"] = it["embed_text"]
        doc = Document(page_content=it["page_content"], metadata=meta)
        docs.append(doc)

    # output folder
    out_dir = os.path.join(FAISS_INDEX_BASE_OUTPUT_DIR, f"{dataset_name}_umls_faiss")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("📡 Building FAISS index...")
    index_wrapper = build_faiss_index(docs, embedding_model)

    print("💾 Saving FAISS index...")
    index_wrapper.save_local(out_dir)

    print(f"✅ Saved to: {out_dir}")

print("\n🎉 ALL DONE — FAISS indexes created.")
