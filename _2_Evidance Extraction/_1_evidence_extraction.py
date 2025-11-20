import os
import json
import re
from collections import defaultdict
import numpy as np
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import logging
import faiss
import torch


# ============================================================
# 1. CONFIG
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evidence_extractor")

CLAIM_INPUT_ROOT = Path("../data/claims/claim_extracted_umls")     # UMLS-normalized claim triplets
REFERENCE_ROOT = Path("../data/claims/processed")                  # For labels and evidence doc IDs
OUTPUT_ROOT = Path("../data/retrievals/docs_kg_structure")

FAISS_INDEXES = [
    # {
    #     "name": "Covid-Fact",
    #     "path": Path("../../data/Primary Evidance Source/_1_doc_as_kg_faiss_indexes_modernbert/Covid-Fact_triplets_flatL2"),
    #     "max_distance": 1.0
    # },
    # {
    #     "name": "HealthVer",
    #     "path": Path("../../data/Primary Evidance Source/_1_doc_as_kg_faiss_indexes_modernbert/HealthVer_triplets_flatL2"),
    #     "max_distance": 1.0
    # },
    {
        "name": "SciFact",
        "path": Path("../data/Primary Evidence Source/umls_triplet_faiss_indexes/SciFact_umls_faiss"),
        "max_distance": 1.0
    }
]

MODEL_NAME = "lightonai/modernbert-embed-large"
TOP_K = 10000
BATCH_SIZE = 64



# ============================================================
# 2. HELPERS
# ============================================================

def clean_text(text: str) -> str:
    text = re.sub(r"[@#]\w+", "", str(text))
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalized_fact_to_query(fact: dict) -> str:
    """
    Converts a UMLS-normalized fact into the embedding-compatible triplet form.
    """
    subj = fact["subject"]["canonical_name"] or fact["subject"]["text"]
    pred = fact["predicate"]
    obj = fact["object"]["canonical_name"] or fact["object"]["text"]

    return f"{subj} [REL] {pred} [TAIL] {obj}"



# ============================================================
# 3. FAISS SEARCH
# ============================================================

def search_across_indexes(vector_stores, queries, top_k, max_distances):
    """
    Batch search across multiple vector stores.
    Returns map: {query_idx: {index_name: [(doc, dist), ...]}}
    """
    results = {}

    if not queries:
        return results

    # embed queries
    embeddings = list(vector_stores.values())[0].embeddings
    query_vecs = np.array(embeddings.embed_documents(queries), dtype=np.float32)

    for i, query in enumerate(queries):
        q_vec_2d = query_vecs[i:i+1]
        results[i] = {}

        for idx_name, store in vector_stores.items():
            max_dist = max_distances[idx_name]
            try:
                dists, indices = store.index.search(q_vec_2d, top_k)

                retrieved = []
                for j in range(indices.shape[1]):
                    pos = int(indices[0][j])
                    dist = float(dists[0][j])

                    if pos < 0: continue
                    if dist > max_dist: continue

                    doc_id = store.index_to_docstore_id.get(pos)
                    if not doc_id: continue

                    doc = store.docstore.search(doc_id)
                    if doc:
                        retrieved.append((doc, dist))

                results[i][idx_name] = retrieved

            except Exception as e:
                logger.error(f"FAISS search error in '{idx_name}': {e}", exc_info=True)
                results[i][idx_name] = []

    return results



# ============================================================
# 4. MAIN RETRIEVAL PIPELINE
# ============================================================

def main():

    # ----------------------------------
    # Load embedding model
    # ----------------------------------
    logger.info("Loading embedding model...")
    try:
        embedder = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={"device": 0 if torch.cuda.is_available() else -1, "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
            cache_folder="../../models/embeddings/"
        )
    except Exception as e:
        logger.error(f"Embedding model load failure: {e}")
        return

    # ----------------------------------
    # Load FAISS indexes
    # ----------------------------------
    vector_stores = {}
    max_distances = {}

    logger.info("\nLoading FAISS indexes...")
    for info in FAISS_INDEXES:
        if not info["path"].exists():
            logger.warning(f"Missing index dir: {info['path']}")
            continue

        try:
            vs = FAISS.load_local(
                folder_path=str(info["path"]),
                embeddings=embedder,
                allow_dangerous_deserialization=True
            )
            vector_stores[info["name"]] = vs
            max_distances[info["name"]] = info.get("max_distance", float("inf"))
            logger.info(f"Loaded index: {info['name']}")

        except Exception as e:
            logger.error(f"Error loading index {info['name']}: {e}")

    if not vector_stores:
        logger.error("No indexes loaded — stopping.")
        return


    # ----------------------------------
    # Iterate over claim datasets
    # ----------------------------------
    for dataset_dir in CLAIM_INPUT_ROOT.iterdir():
        if not dataset_dir.is_dir(): continue

        dataset_name = dataset_dir.name
        logger.info(f"\n===== DATASET: {dataset_name} =====")

        if dataset_name not in vector_stores:
            logger.warning(f"No FAISS index for dataset {dataset_name}. Skipping.")
            continue

        output_dir = OUTPUT_ROOT / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for split in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
            split_umls = "umls_"+split
            input_file = dataset_dir / split_umls
            ref_file = REFERENCE_ROOT / dataset_name / "claims" / split
            print(ref_file)
            if not input_file.exists() or not ref_file.exists():
                logger.warning(f"Missing split {split} for {dataset_name}")
                continue

            # Load reference ground-truth labels
            ref_map = {}
            with open(ref_file, "r", encoding="utf-8") as rf:
                for line in rf:
                    d = json.loads(line)
                    ref_map[d["claim_id"]] = d

            logger.info(f"Processing {split} for {dataset_name}")

            with open(input_file, "r", encoding="utf-8") as f:
                claim_lines = [json.loads(x) for x in f]

            output_path = output_dir / split

            with open(output_path, "w", encoding="utf-8") as out:
                for batch_start in tqdm(range(0, len(claim_lines), BATCH_SIZE)):

                    batch = claim_lines[batch_start:batch_start + BATCH_SIZE]

                    # Build search queries
                    queries = []
                    origin_map = []

                    for idx, claim in enumerate(batch):
                        facts = claim.get("facts", [])

                        if facts:
                            for f in facts:
                                queries.append(normalized_fact_to_query(f))
                                origin_map.append(("fact", idx, f))
                        else:
                            q = clean_text(claim["claim"])
                            queries.append(q)
                            origin_map.append(("claim_text", idx, None))

                    # Run search
                    results_map = search_across_indexes(
                        {dataset_name: vector_stores[dataset_name]},
                        queries,
                        TOP_K,
                        {dataset_name: max_distances[dataset_name]}
                    )

                    # Group results by claim
                    grouped = defaultdict(lambda: {"facts": [], "claim_text": None})

                    for q_idx, (kind, local_idx, fact_item) in enumerate(origin_map):
                        retrieved = results_map.get(q_idx, {}).get(dataset_name, [])

                        if kind == "fact":
                            grouped[local_idx]["facts"].append({
                                "original_fact": fact_item,
                                "retrieved_relations": [
                                    {"page_content": doc.page_content,
                                     "score": dist,
                                     "metadata": doc.metadata}
                                    for doc, dist in retrieved
                                ]
                            })
                        else:
                            grouped[local_idx]["claim_text"] = [
                                {"page_content": doc.page_content,
                                 "score": dist,
                                 "metadata": doc.metadata}
                                for doc, dist in retrieved
                            ]

                    # Write output
                    for i, claim in enumerate(batch):
                        cid = claim["claim_id"]
                        ref = ref_map.get(cid, {})

                        final = {
                            "claim_id": cid,
                            "claim_text": claim["claim"],
                            "label": ref.get("label", "UNKNOWN"),
                            "evidence_doc_ids": ref.get("evidence_doc_ids", []),
                            "retrieved_kg_relations_per_fact": grouped[i]["facts"],
                            "retrieved_documents_for_claim_text": grouped[i]["claim_text"]
                        }
                        out.write(json.dumps(final) + "\n")

            logger.info(f"Saved evidence file → {output_path}")


if __name__ == "__main__":
    main()
