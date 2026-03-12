from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, field, dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from ming.core.redis import RedisDatabase
from ming.extraction.kg_schema import Chunk, Entity, Relationship, CanonicalEntity
from datasketch import MinHash, MinHashLSH

@dataclass
class ERConfig: 
    threshold: float = 0.5
    num_perm: int = 128     

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

        lsh = MinHashLSH(threshold=self.er_config.threshold, num_perm=self.er_config.num_perm)
        minhashes = {}

        for ent in new_entities:
            m = MinHash(num_perm=self.er_config.num_perm)
            for d in set(ent.text.lower().split()):
                m.update(d.encode('utf8'))
            minhashes[ent.entity_id] = m
            lsh.insert(ent.entity_id, m)

        clusters = []
        visited = set()

        for ent in new_entities:
            if ent.entity_id in visited:
                continue
            
            result = lsh.query(minhashes[ent.entity_id])
            cluster_members = [eid for eid in result if eid not in visited]
            final_cluster = []
            for eid in cluster_members:
                target_ent = next((e for e in new_entities if e.entity_id == eid), None)
                if target_ent and target_ent.label == ent.label:
                    final_cluster.append(target_ent)
                    visited.add(eid)
            
            if final_cluster:
                clusters.append(final_cluster)

        canonical_entities = []
        for cluster in clusters:
            canonical_name = max([e.text for e in cluster], key=len)
            canonical_id = f"can_{uuid4().hex}"
            
            merged_rel_ids = list(set([rel_id for e in cluster for rel_id in e.relationships]))
            entity_ids = [e.entity_id for e in cluster]
            
            ce = CanonicalEntity(
                canonical_id=canonical_id,
                text=canonical_name,
                label=cluster[0].label,
                entities=entity_ids,
                relationships=merged_rel_ids
            )
            canonical_entities.append(ce)
            
            for ent in cluster:
                ent.resolved_id = canonical_id
        
        self.save_entities(new_entities)
        self.save_canonical_entities(canonical_entities)
    
    def search_entity_by_id(self, entity_id: str) -> Entity:
        return self.database.get_entry(entity_id)

    def search_entities_by_text(self, text: str) -> List[Entity]:
        return [e for e in self.database.get_entries() if e.text == text]
    
    def search_chunk_by_id(self, chunk_id: str) -> Chunk:
        return self.database.get_entry(chunk_id)
    
    def search_relationship_by_id(self, relationship_id: str) -> Relationship:
        return self.database.get_entry(relationship_id)

    def search_relationship_by_subject(self, subject: str) -> List[Relationship]:
        return [r for r in self.database.get_entries() if r.subject == subject]
    
    def search_relationship_by_object(self, object: str) -> List[Relationship]:
        return [r for r in self.database.get_entries() if r.object == object]

    # ── Query helpers ──────────────────────────────────────────────────

    def _scan_all_entries(
        self,
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict]]:
        """Scan Redis and categorise every KG entry by type."""
        relationships = {}
        entities = {}
        chunks = {}

        cursor = 0
        while True:
            cursor, keys = self.database.client.scan(cursor=cursor, count=200)
            for key in keys:
                if key.startswith(("url:", "url:lock:", "queries:")):
                    continue
                data = self.database.get_entry(key)
                if not data:
                    continue
                if "relationship_id" in data:
                    relationships[key] = data
                elif "entity_id" in data:
                    entities[key] = data
                elif "chunk_id" in data:
                    chunks[key] = data
            if cursor == 0:
                break

        return relationships, entities, chunks

    def _chunk_text_for_relationship(
        self,
        rel_id: str,
        entities: Dict[str, Dict],
        chunks: Dict[str, Dict],
    ) -> str:
        for ent_data in entities.values():
            rels_raw = ent_data.get("relationships", "[]")
            rels = json.loads(rels_raw) if isinstance(rels_raw, str) else rels_raw
            if rel_id in rels:
                chunk_data = chunks.get(ent_data.get("chunk_id", ""))
                if chunk_data:
                    return chunk_data.get("text", "")
        return ""

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
    def _group_by_chunk(pairs: List[Tuple[Dict, str]]) -> List[str]:
        """Consolidate relationships that share the same chunk into one entry.

        Returns one string per unique chunk:
            <rel line 1>
            <rel line 2>
            ...
            <chunk text>
        """
        chunk_to_rels: Dict[str, List[str]] = {}
        chunk_order: List[str] = []

        for rel_data, chunk_text in pairs:
            rel_line = KGRedisStore._format_rel_line(rel_data)
            if chunk_text not in chunk_to_rels:
                chunk_to_rels[chunk_text] = []
                chunk_order.append(chunk_text)
            chunk_to_rels[chunk_text].append(rel_line)

        results: List[str] = []
        for chunk_text in chunk_order:
            rel_lines = "\n".join(chunk_to_rels[chunk_text])
            results.append(f"{rel_lines}\n{chunk_text}")
        return results

    def get_neighbors(self, subject: str) -> List[str]:
        """All relationships whose *subject* matches, grouped by context chunk."""
        relationships, entities, chunks = self._scan_all_entries()
        subject_lower = subject.lower()
        pairs: List[Tuple[Dict, str]] = []

        for rel_data in relationships.values():
            if rel_data.get("subject", "").lower() == subject_lower:
                chunk_text = self._chunk_text_for_relationship(
                    rel_data["relationship_id"], entities, chunks,
                )
                pairs.append((rel_data, chunk_text))

        return self._group_by_chunk(pairs)

    def find_connection(self, subject: str, object_text: str) -> List[str]:
        """BFS shortest path from *subject* to *object_text*, grouped by chunk."""
        relationships, entities, chunks = self._scan_all_entries()

        adj: Dict[str, List[Tuple[Dict, str]]] = {}
        for rel_data in relationships.values():
            src = rel_data.get("subject", "").lower()
            tgt = rel_data.get("object", "").lower()
            adj.setdefault(src, []).append((rel_data, tgt))

        subject_lower = subject.lower()
        object_lower = object_text.lower()

        queue: deque[Tuple[str, List[Dict]]] = deque([(subject_lower, [])])
        visited = {subject_lower}

        while queue:
            current, path = queue.popleft()
            if current == object_lower and path:
                pairs: List[Tuple[Dict, str]] = []
                for rel_data in path:
                    chunk_text = self._chunk_text_for_relationship(
                        rel_data["relationship_id"], entities, chunks,
                    )
                    pairs.append((rel_data, chunk_text))
                return self._group_by_chunk(pairs)

            for rel_data, neighbor in adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [rel_data]))

        return []