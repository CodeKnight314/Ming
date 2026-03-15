from __future__ import annotations

import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from threading import local
from typing import Dict, Iterable, List, Optional, Union

from ming.extraction.ner_module import Chunk, Entity, NERModule
from ming.extraction.re_module import REModule, Relationship
from ming.models import OpenRouterModelConfig
from ming.extraction.selection_policy import (
    calculate_entity_density,
    calculate_source_score,
    chunk_optimization,
)
from ming.extraction.kg_module import KGRedisStore
from ming.extraction.kg_schema import Chunk as KGChunk, Entity as KGEntity

SOURCE_SCORE_CUTOFF = 6.0


@dataclass
class SentenceExtraction:
    sentence: str
    entities: List[Entity]
    relationships: List[Relationship]


@dataclass
class ChunkExtraction:
    """RE extraction result for a single chunk."""

    chunk_text: str
    start: int
    end: int
    entities: List[Entity]
    relationships: List[Relationship]
    url: str


@dataclass
class PipelineResult:
    entities: List[Entity]
    relationships: List[Relationship]
    chunk_extractions: List[ChunkExtraction]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SourceChunkCollection:
    text: str
    url: str
    chunks: List[Chunk]
    source_score: float


class NERREPipeline:
    def __init__(
        self,
        re_config: Union[dict, OpenRouterModelConfig],
        kg_store: KGRedisStore,
        max_workers: int = 4,
    ):
        self.ner = NERModule()
        self.re_config = re_config
        self.kg_store = kg_store
        self.max_workers = max(1, max_workers)
        self._re_local = local()

    def _get_re_module(self) -> REModule:
        module = getattr(self._re_local, "module", None)
        if module is None:
            module = REModule(self.re_config)
            self._re_local.module = module
        return module

    def collect_source_chunks(self, text: str, url: str) -> SourceChunkCollection:
        chunks = self.ner.run(text, url)
        densities = calculate_entity_density(chunks)
        source_score = calculate_source_score(densities)
        return SourceChunkCollection(
            text=text,
            url=url,
            chunks=chunks,
            source_score=source_score,
        )

    def collect_sources(
        self, sources: Iterable[tuple[str, str]]
    ) -> List[SourceChunkCollection]:
        return [self.collect_source_chunks(text, url) for text, url in sources]

    def select_chunks_for_re(
        self,
        sources: List[SourceChunkCollection],
        source_score_cutoff: float = SOURCE_SCORE_CUTOFF,
    ) -> List[Chunk]:
        accepted_chunks: List[Chunk] = []
        for source in sources:
            if source.source_score < source_score_cutoff:
                continue
            accepted_chunks.extend(source.chunks)

        if not accepted_chunks:
            return []

        return chunk_optimization(accepted_chunks)

    def _extract_chunk_relationships(self, chunk: Chunk) -> ChunkExtraction:
        """Extract relationships for a chunk: one API call with chunk text and its entities."""
        target_entities = list(OrderedDict.fromkeys(e.text for e in chunk.entities))
        if not target_entities:
            return ChunkExtraction(
                chunk_text=chunk.text,
                start=chunk.start,
                end=chunk.end,
                entities=chunk.entities,
                relationships=[],
                url=chunk.url,
            )
        relationships = self._get_re_module().run(
            chunk.text, target_entities
        )
        return ChunkExtraction(
            chunk_text=chunk.text,
            start=chunk.start,
            end=chunk.end,
            entities=chunk.entities,
            relationships=relationships,
            url=chunk.url,
        )

    def _dedupe_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        seen = {}
        deduped = []
        for relationship in relationships:
            key = (
                relationship.subject,
                relationship.predicate,
                relationship.object,
                relationship.object_type,
            )
            if key in seen:
                continue
            seen[key] = relationship
            deduped.append(relationship)
        return deduped

    def _build_kg_records(
        self, chunks: List[Chunk]
    ) -> tuple[List[KGChunk], List[KGEntity], Dict[tuple[int, str], List[KGEntity]]]:
        kg_chunks = []
        kg_entities = []
        entity_map: Dict[tuple[int, str], List[KGEntity]] = {}

        for i, chunk in enumerate(chunks):
            chunk_id = uuid.uuid4().hex

            entity_ids = []
            for entity in chunk.entities:
                entity_id = uuid.uuid4().hex
                kg_entity = KGEntity(
                    entity_id=entity_id,
                    text=entity.text,
                    label=entity.label,
                    chunk_id=chunk_id,
                    relationships=[]
                )
                kg_entities.append(kg_entity)
                entity_ids.append(entity_id)

                if (i, entity.text) not in entity_map:
                    entity_map[(i, entity.text)] = []
                entity_map[(i, entity.text)].append(kg_entity)

            kg_chunk = KGChunk(
                chunk_id=chunk_id,
                text=chunk.text,
                entities=entity_ids,
                url=chunk.url,
            )
            kg_chunks.append(kg_chunk)

        return kg_chunks, kg_entities, entity_map

    def run_re_on_chunks(self, chunks: List[Chunk]) -> List[KGEntity]:
        if not chunks:
            return []

        kg_chunks, kg_entities, entity_map = self._build_kg_records(chunks)
        chunks_with_entities_indices = [i for i, c in enumerate(chunks) if c.entities]

        if not chunks_with_entities_indices:
            self.kg_store.save_chunks(kg_chunks)
            self.kg_store.save_entities(kg_entities)
            return kg_entities

        max_workers = min(self.max_workers, len(chunks_with_entities_indices))
        results_by_chunk_idx = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self._extract_chunk_relationships, chunks[idx]): idx
                for idx in chunks_with_entities_indices
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                results_by_chunk_idx[idx] = future.result()

        all_relationships = []
        rel_to_entities = []

        for idx in chunks_with_entities_indices:
            extraction = results_by_chunk_idx.get(idx)
            if not extraction:
                continue
            for rel in extraction.relationships:
                matching_entities = entity_map.get((idx, rel.subject), [])
                for me in matching_entities:
                    rel_to_entities.append((rel, me))
                all_relationships.append(rel)

        final_relationships = self._dedupe_relationships(all_relationships)
        seen_rels = {
            (rel.subject, rel.predicate, rel.object, rel.object_type): rel
            for rel in final_relationships
        }

        for rel, entity in rel_to_entities:
            key = (rel.subject, rel.predicate, rel.object, rel.object_type)
            representative_rel = seen_rels[key]
            if representative_rel.relationship_id not in entity.relationships:
                entity.relationships.append(representative_rel.relationship_id)

        self.kg_store.save_chunks(kg_chunks)
        self.kg_store.save_entities(kg_entities)
        self.kg_store.save_relationships(final_relationships)
        return kg_entities

    def run_batch(
        self,
        sources: Iterable[tuple[str, str]],
        source_score_cutoff: float = SOURCE_SCORE_CUTOFF,
    ) -> List[KGEntity]:
        collected_sources = self.collect_sources(sources)
        optimized_chunks = self.select_chunks_for_re(
            collected_sources,
            source_score_cutoff=source_score_cutoff,
        )
        return self.run_re_on_chunks(optimized_chunks)

    def run(
        self,
        text: str,
        url: str,
        source_score_cutoff: float = SOURCE_SCORE_CUTOFF,
    ) -> List[KGEntity]:
        return self.run_batch(
            [(text, url)],
            source_score_cutoff=source_score_cutoff,
        )