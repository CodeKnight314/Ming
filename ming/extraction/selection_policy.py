from ming.extraction.ner_module import Chunk
from math import log
from collections import Counter
from typing import List, Set
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np
import networkx as nx
from community import community_louvain
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

def tokenize(text: str):
    return text.split()


def unique_entities(chunk: Chunk) -> Set[str]:
    return set(e.text.lower() for e in chunk.entities)


def entity_entropy(chunk: Chunk) -> float:
    if not chunk.entities:
        return 0.0

    counts = Counter(e.text.lower() for e in chunk.entities)
    total = sum(counts.values())

    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * log(p + 1e-9)

    return entropy


def relationship_potential(num_entities: int) -> float:
    if num_entities < 2:
        return 0.0

    pairs = num_entities * (num_entities - 1) / 2

    return log(1 + pairs)


def entity_density(chunk: Chunk) -> float:
    tokens = tokenize(chunk.text)
    token_count = len(tokens)

    if token_count == 0:
        return 0.0

    unique_ents = unique_entities(chunk)

    return len(unique_ents) / log(token_count + 1)


def novelty_score(chunk: Chunk, other_chunks: List[Chunk]) -> float:
    ents = unique_entities(chunk)
    if not ents:
        return 1.0

    max_overlap = 0.0

    for other in other_chunks:
        if other is chunk:
            continue

        other_ents = unique_entities(other)

        if not other_ents:
            continue

        intersection = len(ents & other_ents)
        union = len(ents | other_ents)

        overlap = intersection / union

        max_overlap = max(max_overlap, overlap)

    return 1 - max_overlap


def calculate_entity_density(
    chunks: List[Chunk],
    density_coeff: float = 0.4,
    relationship_coeff: float = 0.3,
    entropy_coeff: float = 0.2,
    novelty_coeff: float = 0.1
) -> List[float]:

    if abs(density_coeff + relationship_coeff + entropy_coeff + novelty_coeff - 1.0) > 1e-6:
        raise ValueError("coefficients must sum to 1")

    scores = []
    for chunk in chunks:
        number_of_entities = len(unique_entities(chunk))
        density = entity_density(chunk)
        relation_score = relationship_potential(number_of_entities)
        entropy = entity_entropy(chunk)
        novelty = novelty_score(chunk, chunks)

        score = (
            density_coeff * density
            + relationship_coeff * relation_score
            + entropy_coeff * entropy
            + novelty_coeff * novelty
        )
        scores.append(score)

    return scores

def calculate_source_score(scores: List[float], k: int = 10) -> float:
    if not scores:
        return 0.0

    top = sorted(scores, reverse=True)[:k]
    weights = [1.0 / (i + 1) for i in range(len(top))]
    weighted_sum = sum(w * s for w, s in zip(weights, top))
    return weighted_sum

def jaccard_sim(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def redundancy_score(chunk_a, chunk_b) -> float:
    # 1. Semantic (keep)
    emb_cos = np.dot(chunk_a.embedding, chunk_b.embedding) / (
        np.linalg.norm(chunk_a.embedding) * np.linalg.norm(chunk_b.embedding)
    )
    
    # 2. Lexical (keep, but optional: switch to ngram_range=(1,2) in TfidfVectorizer)
    tfidf_cos = np.dot(chunk_a.tfidf_embedding, chunk_b.tfidf_embedding) / (
        np.linalg.norm(chunk_a.tfidf_embedding) * np.linalg.norm(chunk_b.tfidf_embedding)
    )
    
    # 3. Token Jaccard (new – catches paraphrases)
    tokens_a = set(chunk_a.text.lower().split())  # or proper tokenizer + stopword removal
    tokens_b = set(chunk_b.text.lower().split())
    token_jac = jaccard_sim(tokens_a, tokens_b)
    
    # 4. Entity Jaccard (new – huge win for history/research chunks; you already extract .entities)
    ent_jac = jaccard_sim(unique_entities(chunk_a), unique_entities(chunk_b))
    
    # Weighted composite – tuned so pure paraphrases jump >0.65 while complementary stays <0.48
    score = (0.35 * emb_cos +
             0.15 * tfidf_cos +
             0.40 * token_jac +
             0.15 * ent_jac)
    
    return score

def chunk_optimization(chunks: List[Chunk], k: int = 20, sim_threshold: float = 0.40) -> List[Chunk]:
    if len(chunks) <= 1:
        return list(chunks)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    tfidf_model = TfidfVectorizer()
    
    corpus = [chunk.text for chunk in chunks]
    tfidf_model.fit(corpus)
    tfidf_matrix = tfidf_model.transform(corpus)
    
    embeddings = model.encode(corpus, batch_size=64)
    
    for i, chunk in enumerate(chunks):
        chunk.embedding = embeddings[i]
        chunk.tfidf_embedding = tfidf_matrix[i].toarray().flatten()

    vectors = np.array([c.embedding for c in chunks], dtype=np.float32)
    faiss.normalize_L2(vectors)
    
    dim = vectors.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 128
    index.hnsw.efSearch = max(64, k + 1)
    index.add(vectors)
    
    neighbor_count = min(k + 1, len(chunks))
    distances, neighbors = index.search(vectors, neighbor_count)

    candidate_pairs = {}
    for i in range(len(chunks)):
        for pos in range(1, neighbor_count):
            j = int(neighbors[i][pos])
            sim = float(distances[i][pos])
            if sim >= sim_threshold:
                pair = (min(i, j), max(i, j))
                if pair not in candidate_pairs or sim > candidate_pairs[pair]:
                    candidate_pairs[pair] = sim

    pair_scores = {}
    scores = []
    for (i, j) in candidate_pairs:
        score = redundancy_score(chunks[i], chunks[j])
        scores.append(score)
        if score >= sim_threshold:
            pair_scores[(i, j)] = score

    G = nx.Graph()
    G.add_nodes_from(range(len(chunks)))
    for (i, j), score in pair_scores.items():
        G.add_edge(i, j, weight=score)

    partition = community_louvain.best_partition(G, weight='weight')
    clusters = {}
    for chunk_idx, cluster_id in partition.items():
        clusters.setdefault(cluster_id, []).append(chunk_idx)

    representatives = set()
    
    for cluster_id, member_indices in clusters.items():
        if len(member_indices) == 1:
            representatives.add(member_indices[0])
            continue
        
        best_idx = None
        best_score = -1.0
        
        cluster_vectors = np.array([chunks[i].embedding for i in member_indices])
        centroid = cluster_vectors.mean(axis=0)
        
        for idx in member_indices:
            centrality = float(np.dot(chunks[idx].embedding, centroid) / (
                np.linalg.norm(chunks[idx].embedding) * np.linalg.norm(centroid)
            ))
            
            length_score = min(len(chunks[idx].text) / 1500, 1.0)
            
            entity_score = min(len(chunks[idx].entities) / 10, 1.0)
            
            combined = 0.4 * centrality + 0.35 * length_score + 0.25 * entity_score
            
            if combined > best_score:
                best_score = combined
                best_idx = idx
        
        representatives.add(best_idx)

    deduplicated = [chunks[i] for i in sorted(representatives)]
    return deduplicated