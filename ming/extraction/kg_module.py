from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, get_origin
from uuid import uuid4

from ming.core.redis import RedisDatabase
from ming.core.reference_cleanup import canonicalize_url
from ming.core.text_metrics import normalize_whitespace, tokenize_for_overlap
from ming.extraction.kg_schema import Chunk, Entity, Relationship, CanonicalEntity
from datasketch import MinHash, MinHashLSH
import re
from collections import Counter
import numpy as np
import faiss 
from sentence_transformers import SentenceTransformer

T = TypeVar("T")

@dataclass
class ERConfig: 
    threshold: float = 0.5
    num_perm: int = 128     
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_threshold: float = 0.78
    embedding_top_k: int = 12

class KGRedisStore:
    def __init__(self, database: RedisDatabase, er_config: ERConfig):
        self.database = database
        self.er_config = er_config

    def save_entities(self, entities: List[Entity]) -> None:
        for entity in entities:
            self.database.update_entry(entity.entity_id, asdict(entity))

    def save_chunks(self, chunks: List[Chunk]) -> None:
        for chunk in chunks:
            self.database.update_entry(chunk.chunk_id, asdict(chunk))

    def save_relationships(self, relationships: List[Relationship]) -> None:
        for relationship in relationships:
            self.database.update_entry(relationship.relationship_id, asdict(relationship))

    def save_canonical_entities(self, canonical_entities: List[CanonicalEntity]) -> None:
        for ce in canonical_entities:
            self.database.update_entry(ce.canonical_id, asdict(ce))

    def perform_entity_resolution(self, new_entities: List[Entity]) -> None:
        if not new_entities:
            return

        def _normalize_entity_text(text: str) -> str:
            return normalize_whitespace(text or "").strip()

        def _labels_compatible(a: str, b: str) -> bool:
            if not a or not b:
                return False
            if a == b:
                return True
            compatible = {
                ("ORG", "PRODUCT"),
                ("PRODUCT", "ORG"),
                ("GPE", "LOC"),
                ("LOC", "GPE"),
            }
            return (a, b) in compatible

        def _minhash_tokens(text: str) -> set[str]:
            t = _normalize_entity_text(text).lower()
            if not t:
                return set()
            parts = t.split()
            if len(parts) > 1:
                return set(parts)
            # single token: use character 3-grams (better for typos/variations than whole-token Jaccard)
            compact = re.sub(r"\s+", "", t)
            if len(compact) <= 3:
                return {compact}
            return {compact[i : i + 3] for i in range(0, len(compact) - 2)}

        def _canonical_name_choice(texts: List[str]) -> str:
            # Prefer natural names over bloated or acronym-only strings.
            candidates = [_normalize_entity_text(t) for t in texts if _normalize_entity_text(t)]
            if not candidates:
                return ""

            def score(name: str) -> tuple[float, float, float]:
                n = name.strip()
                lower = n.lower()
                length = len(n)
                words = n.split()
                has_space = 1.0 if len(words) >= 2 else 0.0
                is_acronym = 1.0 if (n.isupper() and length <= 6) else 0.0
                starts_with_the = 1.0 if lower.startswith("the ") else 0.0
                # Prefer moderate length; penalize very long names.
                ideal = 18.0
                length_term = -abs(length - ideal) / ideal
                long_penalty = -max(0.0, (length - 32) / 32)
                return (
                    has_space * 2.0 + length_term + long_penalty - is_acronym * 1.5 - starts_with_the * 0.8,
                    -is_acronym,
                    -length,
                )

            # Best score wins; tiebreaker favors shorter (less bloated) via score's last term.
            return max(candidates, key=score)

        entities_by_id: Dict[str, Entity] = {entity.entity_id: entity for entity in new_entities}
        ids = [e.entity_id for e in new_entities]

        # Union-Find for merges
        parent: Dict[str, str] = {eid: eid for eid in ids}
        rank: Dict[str, int] = {eid: 0 for eid in ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        # Phase 1: MinHash LSH
        lsh = MinHashLSH(threshold=self.er_config.threshold, num_perm=self.er_config.num_perm)
        minhashes: Dict[str, MinHash] = {}

        for ent in new_entities:
            m = MinHash(num_perm=self.er_config.num_perm)
            for token in _minhash_tokens(ent.text):
                if token:
                    m.update(token.encode("utf8"))
            minhashes[ent.entity_id] = m
            lsh.insert(ent.entity_id, m)

        for ent in new_entities:
            hits = lsh.query(minhashes[ent.entity_id])
            for other_id in hits:
                if other_id == ent.entity_id:
                    continue
                other = entities_by_id.get(other_id)
                if other is None:
                    continue
                if _labels_compatible(ent.label, other.label):
                    union(ent.entity_id, other_id)

        # Phase 2: Embedding-based merges (abbreviations, acronyms, semantic equivalents)
        if (
            SentenceTransformer is not None
            and np is not None
            and faiss is not None
            and float(self.er_config.embedding_threshold) > 0.0
        ):
            # Build representative texts per current UF component.
            rep_to_ids: Dict[str, List[str]] = {}
            for eid in ids:
                rep_to_ids.setdefault(find(eid), []).append(eid)

            reps: List[str] = list(rep_to_ids.keys())
            rep_texts: List[str] = []
            rep_labels: List[str] = []
            for rep in reps:
                member_texts = [entities_by_id[eid].text for eid in rep_to_ids[rep]]
                rep_texts.append(_canonical_name_choice(member_texts) or entities_by_id[rep_to_ids[rep][0]].text)
                labels = [entities_by_id[eid].label for eid in rep_to_ids[rep] if entities_by_id[eid].label]
                rep_labels.append(Counter(labels).most_common(1)[0][0] if labels else "")

            try:
                model = SentenceTransformer(self.er_config.embedding_model_name, local_files_only=True)
                emb = model.encode(rep_texts, batch_size=128, normalize_embeddings=True)
                vectors = np.array(emb, dtype="float32")

                dim = int(vectors.shape[1])
                index = faiss.IndexFlatIP(dim)
                index.add(vectors)

                top_k = max(2, int(self.er_config.embedding_top_k))
                scores, neighbors = index.search(vectors, min(top_k, len(reps)))

                threshold = float(self.er_config.embedding_threshold)
                for i, rep_i in enumerate(reps):
                    label_i = rep_labels[i]
                    for pos in range(1, neighbors.shape[1]):
                        j = int(neighbors[i][pos])
                        if j < 0 or j == i:
                            continue
                        sim = float(scores[i][pos])
                        if sim < threshold:
                            continue
                        rep_j = reps[j]
                        label_j = rep_labels[j]
                        if _labels_compatible(label_i, label_j):
                            union(rep_i, rep_j)
            except Exception:
                # Embedding pass is best-effort; fall back to MinHash-only.
                pass

        # Materialize clusters from UF
        clusters_by_rep: Dict[str, List[Entity]] = {}
        for eid, ent in entities_by_id.items():
            clusters_by_rep.setdefault(find(eid), []).append(ent)

        canonical_entities: List[CanonicalEntity] = []
        for cluster in clusters_by_rep.values():
            if not cluster:
                continue

            canonical_name = _canonical_name_choice([e.text for e in cluster])
            canonical_id = f"can_{uuid4().hex}"

            merged_rel_ids = list({rel_id for e in cluster for rel_id in (e.relationships or [])})
            entity_ids = [e.entity_id for e in cluster]
            labels = [e.label for e in cluster if e.label]
            canonical_label = Counter(labels).most_common(1)[0][0] if labels else cluster[0].label

            ce = CanonicalEntity(
                canonical_id=canonical_id,
                text=canonical_name,
                label=canonical_label,
                entities=entity_ids,
                relationships=merged_rel_ids,
            )
            canonical_entities.append(ce)

            for ent in cluster:
                ent.resolved_id = canonical_id

        self.save_entities(new_entities)
        self.save_canonical_entities(canonical_entities)

    def _deserialize_entry(self, data: Dict[str, Any], model: Type[T]) -> T:
        parsed: Dict[str, Any] = {}
        for schema_field in fields(model):
            value = data.get(schema_field.name)
            if value is None:
                continue
            if get_origin(schema_field.type) is list:
                parsed[schema_field.name] = json.loads(value) if isinstance(value, str) and value else []
            elif schema_field.type is float:
                parsed[schema_field.name] = float(value) if value not in ("", None) else 0.0
            else:
                parsed[schema_field.name] = value
        return model(**parsed)
    
    def search_entity_by_id(self, entity_id: str) -> Entity:
        return self._deserialize_entry(self.database.get_entry(entity_id), Entity)

    def search_entities_by_text(self, text: str) -> List[Entity]:
        _, entities, _, _ = self._scan_all_entries()
        text_lower = text.lower()
        return [
            self._deserialize_entry(entity_data, Entity)
            for entity_data in entities.values()
            if entity_data.get("text", "").lower() == text_lower
        ]
    
    def search_chunk_by_id(self, chunk_id: str) -> Chunk:
        return self._deserialize_entry(self.database.get_entry(chunk_id), Chunk)
    
    def search_relationship_by_id(self, relationship_id: str) -> Relationship:
        return self._deserialize_entry(self.database.get_entry(relationship_id), Relationship)

    def search_relationship_by_subject(self, subject: str) -> List[Relationship]:
        relationships, _, _, _ = self._scan_all_entries()
        subject_lower = subject.lower()
        return [
            self._deserialize_entry(rel_data, Relationship)
            for rel_data in relationships.values()
            if rel_data.get("subject", "").lower() == subject_lower
        ]
    
    def search_relationship_by_object(self, object: str) -> List[Relationship]:
        relationships, _, _, _ = self._scan_all_entries()
        object_lower = object.lower()
        return [
            self._deserialize_entry(rel_data, Relationship)
            for rel_data in relationships.values()
            if rel_data.get("object", "").lower() == object_lower
        ]

    def get_entities(self) -> List[Entity]:
        _, entities, _, _ = self._scan_all_entries()
        return [self._deserialize_entry(entity_data, Entity) for entity_data in entities.values()]

    def get_relationships(self) -> List[Relationship]:
        relationships, _, _, _ = self._scan_all_entries()
        return [self._deserialize_entry(rel_data, Relationship) for rel_data in relationships.values()]

    def get_chunks(self) -> List[Chunk]:
        _, _, chunks, _ = self._scan_all_entries()
        return [self._deserialize_entry(chunk_data, Chunk) for chunk_data in chunks.values()]

    def get_canonical_entities(self) -> List[CanonicalEntity]:
        _, _, _, canonical_entities = self._scan_all_entries()
        return [
            self._deserialize_entry(canonical_entity_data, CanonicalEntity)
            for canonical_entity_data in canonical_entities.values()
        ]

    # ── Query helpers ──────────────────────────────────────────────────

    def _scan_all_entries(
        self,
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict], Dict[str, Dict]]:
        """Scan Redis and categorise every KG entry by type."""
        relationships = {}
        entities = {}
        chunks = {}
        canonical_entities = {}

        cursor = 0
        while True:
            cursor, keys = self.database.client.scan(cursor=cursor, count=200)
            for key in keys:
                if key.startswith(("url:", "url:lock:", "queries:")):
                    continue
                data = self.database.get_entry(key)
                if not data:
                    continue
                if "canonical_id" in data:
                    canonical_entities[key] = data
                elif "relationship_id" in data:
                    relationships[key] = data
                elif "entity_id" in data:
                    entities[key] = data
                elif "chunk_id" in data:
                    chunks[key] = data
            if cursor == 0:
                break

        return relationships, entities, chunks, canonical_entities

    def _find_canonical_entity(self, text: str, canonical_entities: Dict[str, Dict]) -> Optional[Dict[str, Any]]:
        """Find a canonical entity whose name matches (case-insensitive) the query text."""
        text_lower = text.lower()
        for ce in canonical_entities.values():
            if ce.get("text", "").lower() == text_lower:
                return ce
        return None

    def _chunk_text_for_relationship(
        self,
        rel_id: str,
        entities: Dict[str, Dict],
        chunks: Dict[str, Dict],
    ) -> Tuple[str, str]:
        """Returns (chunk_text, url) for a relationship."""
        for ent_data in entities.values():
            rels_raw = ent_data.get("relationships", "[]")
            rels = json.loads(rels_raw) if isinstance(rels_raw, str) else rels_raw
            if rel_id in rels:
                chunk_data = chunks.get(ent_data.get("chunk_id", ""))
                if chunk_data:
                    return chunk_data.get("text", ""), chunk_data.get("url", "")
        return "", ""

    @staticmethod
    def _normalize_predicate(predicate: str) -> str:
        return normalize_whitespace(predicate).lower()

    @staticmethod
    def _truncate_excerpt(text: str, max_chars: int = 280) -> str:
        cleaned = normalize_whitespace(text)
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + "..."

    def _build_resolved_name_lookup(
        self,
        entities: Dict[str, Dict],
        canonical_entities: Dict[str, Dict],
    ) -> Dict[str, str]:
        resolved: Dict[str, str] = {}
        for entity_data in entities.values():
            raw_text = normalize_whitespace(entity_data.get("text", ""))
            if not raw_text:
                continue
            resolved_id = entity_data.get("resolved_id", "")
            canonical_name = ""
            if resolved_id:
                canonical_name = normalize_whitespace(
                    canonical_entities.get(resolved_id, {}).get("text", "")
                )
            if canonical_name:
                resolved.setdefault(raw_text.lower(), canonical_name)
        return resolved

    def _canonical_name(
        self,
        text: str,
        *,
        canonical_entities: Dict[str, Dict],
        resolved_lookup: Dict[str, str],
    ) -> str:
        normalized = normalize_whitespace(text)
        if not normalized:
            return ""
        direct = self._find_canonical_entity(normalized, canonical_entities)
        if direct:
            return normalize_whitespace(direct.get("text", "")) or normalized
        return resolved_lookup.get(normalized.lower(), normalized)

    @staticmethod
    def _query_overlap_score(query_tokens: set[str], *texts: str) -> int:
        if not query_tokens:
            return 0
        haystack_tokens: set[str] = set()
        for text in texts:
            haystack_tokens.update(tokenize_for_overlap(text))
        return len(query_tokens & haystack_tokens)

    @staticmethod
    def _format_rel_line(rel_data: Dict[str, str]) -> str:
        subj = rel_data.get("subject", "")
        pred = rel_data.get("predicate", "")
        obj = rel_data.get("object", "")
        obj_type = rel_data.get("object_type", "")

        line = f"{subj} -[{pred}]-> {obj}"
        if obj_type:
            line += f" ({obj_type})"
        return line

    @staticmethod
    def _group_by_chunk(pairs: List[Tuple[Dict, str, str]]) -> List[str]:
        """Consolidate relationships that share the same chunk into one entry.

        Returns one string per unique chunk:
            <rel line 1>
            <rel line 2>
            ...
            <chunk text>
            URL: <url>
        """
        # key is (chunk_text, url)
        chunk_to_rels: Dict[Tuple[str, str], List[str]] = {}
        chunk_order: List[Tuple[str, str]] = []

        for rel_data, chunk_text, url in pairs:
            rel_line = KGRedisStore._format_rel_line(rel_data)
            key = (chunk_text, url)
            if key not in chunk_to_rels:
                chunk_to_rels[key] = []
                chunk_order.append(key)
            chunk_to_rels[key].append(rel_line)

        results: List[str] = []
        for chunk_text, url in chunk_order:
            rel_lines = "\n".join(chunk_to_rels[(chunk_text, url)])
            formatted_url = canonicalize_url(url) if url else url
            results.append(f"{rel_lines}\n{chunk_text}\nURL: {formatted_url}")
        return results

    def get_neighbors(self, subject: str) -> List[str]:
        """All relationships for a canonical entity or raw string match, grouped by chunk."""
        relationships, entities, chunks, canonical_entities = self._scan_all_entries()
        
        # 1. Try to find a canonical entity for this subject
        ce = self._find_canonical_entity(subject, canonical_entities)
        
        target_rel_ids = set()
        if ce:
            rel_ids_raw = ce.get("relationships", "[]")
            rel_ids = json.loads(rel_ids_raw) if isinstance(rel_ids_raw, str) else rel_ids_raw
            target_rel_ids.update(rel_ids)
        
        # 2. Collect pairs (rel_data, chunk_text, url)
        pairs: List[Tuple[Dict, str, str]] = []
        subject_lower = subject.lower()

        for rel_id, rel_data in relationships.items():
            # Match if ID is in canonical list OR if raw subject matches
            if rel_id in target_rel_ids or rel_data.get("subject", "").lower() == subject_lower:
                chunk_text, url = self._chunk_text_for_relationship(
                    rel_data["relationship_id"], entities, chunks,
                )
                pairs.append((rel_data, chunk_text, url))

        return self._group_by_chunk(pairs)

    def find_connection(self, subject: str, object_text: str) -> List[str]:
        """BFS shortest path using canonical entities when available."""
        relationships, entities, chunks, canonical_entities = self._scan_all_entries()

        # Build adjacency: node -> list of (rel_data, next_node_text)
        adj: Dict[str, List[Tuple[Dict, str]]] = {}
        for rel_id, rel_data in relationships.items():
            src = rel_data.get("subject", "").lower()
            tgt = rel_data.get("object", "").lower()
            
            # If subject is resolved, use canonical name for more connections
            ce_src = self._find_canonical_entity(src, canonical_entities)
            src_key = ce_src["text"].lower() if ce_src else src
            
            ce_tgt = self._find_canonical_entity(tgt, canonical_entities)
            tgt_key = ce_tgt["text"].lower() if ce_tgt else tgt

            adj.setdefault(src_key, []).append((rel_data, tgt_key))

        # Resolve starting and ending nodes
        ce_start = self._find_canonical_entity(subject, canonical_entities)
        start_key = ce_start["text"].lower() if ce_start else subject.lower()
        
        ce_end = self._find_canonical_entity(object_text, canonical_entities)
        end_key = ce_end["text"].lower() if ce_end else object_text.lower()

        queue: deque[Tuple[str, List[Dict]]] = deque([(start_key, [])])
        visited = {start_key}

        while queue:
            current, path = queue.popleft()
            if current == end_key and path:
                pairs: List[Tuple[Dict, str, str]] = []
                for rel_data in path:
                    chunk_text, url = self._chunk_text_for_relationship(
                        rel_data["relationship_id"], entities, chunks,
                    )
                    pairs.append((rel_data, chunk_text, url))
                return self._group_by_chunk(pairs)

            for rel_data, neighbor in adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [rel_data]))

        return []

    def search_evidence(
        self,
        query: str,
        limit: int = 10,
        diversify_by_url: bool = True,
    ) -> Dict[str, Any]:
        relationships, entities, chunks, canonical_entities = self._scan_all_entries()
        if not relationships:
            return {
                "query": query,
                "limit": limit,
                "thin_pool": True,
                "unique_url_count": 0,
                "cards": [],
            }

        resolved_lookup = self._build_resolved_name_lookup(entities, canonical_entities)
        rel_to_chunk: Dict[str, Dict[str, Any]] = {}
        for entity_data in entities.values():
            rel_ids_raw = entity_data.get("relationships", "[]")
            rel_ids = json.loads(rel_ids_raw) if isinstance(rel_ids_raw, str) else rel_ids_raw
            chunk_id = entity_data.get("chunk_id", "")
            chunk_data = chunks.get(chunk_id)
            if not chunk_data:
                continue
            for rel_id in rel_ids:
                existing = rel_to_chunk.get(rel_id)
                if existing is None or float(chunk_data.get("chunk_score", 0.0) or 0.0) > float(
                    existing.get("chunk_score", 0.0) or 0.0
                ):
                    rel_to_chunk[rel_id] = chunk_data

        query_tokens = set(tokenize_for_overlap(query))
        grouped: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}

        for rel_data in relationships.values():
            chunk_data = rel_to_chunk.get(rel_data.get("relationship_id", ""))
            if not chunk_data:
                continue

            canonical_subject = self._canonical_name(
                rel_data.get("subject", ""),
                canonical_entities=canonical_entities,
                resolved_lookup=resolved_lookup,
            )
            canonical_object = self._canonical_name(
                rel_data.get("object", ""),
                canonical_entities=canonical_entities,
                resolved_lookup=resolved_lookup,
            ) or normalize_whitespace(rel_data.get("object", ""))
            predicate_norm = self._normalize_predicate(rel_data.get("predicate", ""))
            object_type = normalize_whitespace(rel_data.get("object_type", "")) or "attribute"

            fact_key = (canonical_subject, predicate_norm, canonical_object, object_type)
            fact_text = f"{canonical_subject} {predicate_norm} {canonical_object}".strip()
            excerpt = self._truncate_excerpt(chunk_data.get("text", ""))
            overlap_score = self._query_overlap_score(query_tokens, fact_text, excerpt)

            if query_tokens and overlap_score == 0:
                continue

            raw_url = normalize_whitespace(chunk_data.get("url", ""))
            if not raw_url:
                continue
            canonical_url = canonicalize_url(raw_url)
            chunk_score = float(chunk_data.get("chunk_score", 0.0) or 0.0)
            source_score = float(chunk_data.get("source_score", 0.0) or 0.0)
            confidence = float(rel_data.get("confidence", 0.0) or 0.0)

            group = grouped.setdefault(
                fact_key,
                {
                    "fact": f"{canonical_subject} -[{predicate_norm}]-> {canonical_object} ({object_type})",
                    "support_by_url": {},
                    "query_relevance": 0,
                    "confidence_sum": 0.0,
                    "confidence_count": 0,
                    "best_source_score": 0.0,
                },
            )
            group["query_relevance"] = max(group["query_relevance"], overlap_score)
            group["confidence_sum"] += confidence
            group["confidence_count"] += 1
            group["best_source_score"] = max(group["best_source_score"], source_score)

            existing_support = group["support_by_url"].get(canonical_url)
            support_payload = {
                "url": canonical_url,
                "excerpt": excerpt,
                "chunk_score": chunk_score,
                "source_score": source_score,
                "confidence": confidence,
            }
            if existing_support is None or (
                chunk_score,
                source_score,
            ) > (
                existing_support["chunk_score"],
                existing_support["source_score"],
            ):
                group["support_by_url"][canonical_url] = support_payload

        if not grouped:
            for rel_data in relationships.values():
                chunk_data = rel_to_chunk.get(rel_data.get("relationship_id", ""))
                if not chunk_data:
                    continue
                canonical_subject = self._canonical_name(
                    rel_data.get("subject", ""),
                    canonical_entities=canonical_entities,
                    resolved_lookup=resolved_lookup,
                )
                canonical_object = self._canonical_name(
                    rel_data.get("object", ""),
                    canonical_entities=canonical_entities,
                    resolved_lookup=resolved_lookup,
                ) or normalize_whitespace(rel_data.get("object", ""))
                predicate_norm = self._normalize_predicate(rel_data.get("predicate", ""))
                object_type = normalize_whitespace(rel_data.get("object_type", "")) or "attribute"
                fact_key = (canonical_subject, predicate_norm, canonical_object, object_type)
                raw_url = normalize_whitespace(chunk_data.get("url", ""))
                if not raw_url:
                    continue
                canonical_url = canonicalize_url(raw_url)
                chunk_score = float(chunk_data.get("chunk_score", 0.0) or 0.0)
                source_score = float(chunk_data.get("source_score", 0.0) or 0.0)
                grouped.setdefault(
                    fact_key,
                    {
                        "fact": f"{canonical_subject} -[{predicate_norm}]-> {canonical_object} ({object_type})",
                        "support_by_url": {
                            canonical_url: {
                                "url": canonical_url,
                                "excerpt": self._truncate_excerpt(chunk_data.get("text", "")),
                                "chunk_score": chunk_score,
                                "source_score": source_score,
                                "confidence": float(rel_data.get("confidence", 0.0) or 0.0),
                            }
                        },
                        "query_relevance": 0,
                        "confidence_sum": float(rel_data.get("confidence", 0.0) or 0.0),
                        "confidence_count": 1,
                        "best_source_score": source_score,
                    },
                )
                if len(grouped) >= max(limit * 3, 16):
                    break

        unique_urls = {
            url
            for group in grouped.values()
            for url in group["support_by_url"].keys()
        }
        cards: List[Dict[str, Any]] = []
        for group in grouped.values():
            supports = sorted(
                group["support_by_url"].values(),
                key=lambda item: (
                    item["source_score"],
                    item["chunk_score"],
                    item["url"],
                ),
                reverse=True,
            )
            avg_confidence = (
                group["confidence_sum"] / group["confidence_count"]
                if group["confidence_count"]
                else 0.0
            )
            cards.append(
                {
                    "fact": group["fact"],
                    "supporting_urls": [support["url"] for support in supports],
                    "chunks": [
                        {"url": support["url"], "excerpt": support["excerpt"]}
                        for support in supports
                    ],
                    "support_count": len(supports),
                    "query_relevance": group["query_relevance"],
                    "best_source_score": group["best_source_score"],
                    "avg_confidence": avg_confidence,
                }
            )

        cards.sort(
            key=lambda card: (
                card["query_relevance"],
                card["support_count"],
                card["best_source_score"],
                card["fact"],
            ),
            reverse=True,
        )

        if diversify_by_url and cards:
            buckets: Dict[str, List[Dict[str, Any]]] = {}
            bucket_order: List[str] = []
            for card in cards:
                dominant_url = card["supporting_urls"][0] if card["supporting_urls"] else "__no_url__"
                if dominant_url not in buckets:
                    buckets[dominant_url] = []
                    bucket_order.append(dominant_url)
                buckets[dominant_url].append(card)

            diversified: List[Dict[str, Any]] = []
            while len(diversified) < len(cards):
                progressed = False
                for dominant_url in bucket_order:
                    bucket = buckets[dominant_url]
                    if not bucket:
                        continue
                    diversified.append(bucket.pop(0))
                    progressed = True
                if not progressed:
                    break
            cards = diversified

        limited_cards = cards[: max(1, limit)]
        return {
            "query": query,
            "limit": limit,
            "thin_pool": len(unique_urls) < 8,
            "unique_url_count": len(unique_urls),
            "cards": limited_cards,
        }
