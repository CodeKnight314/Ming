from __future__ import annotations

import json
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
