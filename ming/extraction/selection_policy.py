from ming.extraction import Chunk
from math import log
from collections import Counter
from typing import List, Set
import glob


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