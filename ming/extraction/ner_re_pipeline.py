from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from threading import local
from typing import List, Union

from ming.extraction.ner_module import Chunk, Entity, NERModule
from ming.extraction.re_module import REModule, RERunResult, REUsage, Relationship
from ming.models import OpenRouterModelConfig
from ming.extraction.selection_policy import calculate_source_score, calculate_entity_density


@dataclass
class SentenceExtraction:
    sentence: str
    entities: List[Entity]
    relationships: List[Relationship]
    usage: REUsage


@dataclass
class ChunkExtraction:
    """RE extraction result for a single chunk."""

    chunk_text: str
    start: int
    end: int
    entities: List[Entity]
    relationships: List[Relationship]
    usage: REUsage


@dataclass
class PipelineResult:
    entities: List[Entity]
    relationships: List[Relationship]
    chunk_extractions: List[ChunkExtraction]
    usage: REUsage

    def to_dict(self) -> dict:
        return asdict(self)


class NERREPipeline:
    def __init__(
        self,
        re_config: Union[dict, OpenRouterModelConfig],
        ner_model_name: str = "en_core_web_sm",
        max_workers: int = 4,
    ):
        self.ner = NERModule(model_name=ner_model_name)
        self.re_config = re_config
        self.max_workers = max(1, max_workers)
        self._re_local = local()

    def _get_re_module(self) -> REModule:
        module = getattr(self._re_local, "module", None)
        if module is None:
            module = REModule(self.re_config)
            self._re_local.module = module
        return module

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
                usage=REUsage(),
            )
        re_result: RERunResult = self._get_re_module().run_with_metadata(
            chunk.text, target_entities
        )
        return ChunkExtraction(
            chunk_text=chunk.text,
            start=chunk.start,
            end=chunk.end,
            entities=chunk.entities,
            relationships=re_result.relationships,
            usage=re_result.usage,
        )

    def _dedupe_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        seen = set()
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
            seen.add(key)
            deduped.append(relationship)
        return deduped

    def run(self, text: str) -> PipelineResult:
        chunks = self.ner.run(text)
        densities = calculate_entity_density(chunks)
        source_score = calculate_source_score(densities)
        entities = [e for c in chunks for e in c.entities]

        # Filter to chunks that have entities (skip empty chunks for RE)
        chunks_with_entities = [c for c in chunks if c.entities]

        if not chunks_with_entities or source_score < 4.5:
            return PipelineResult(
                entities=entities,
                relationships=[],
                chunk_extractions=[],
                usage=REUsage(),
            )

        max_workers = min(self.max_workers, len(chunks_with_entities))
        results_by_chunk = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self._extract_chunk_relationships, chunk): id(chunk)
                for chunk in chunks_with_entities
            }
            for future in as_completed(future_map):
                chunk_id = future_map[future]
                results_by_chunk[chunk_id] = future.result()

        ordered_chunk_extractions = [
            results_by_chunk[id(c)] for c in chunks_with_entities
        ]
        all_relationships = self._dedupe_relationships(
            [
                rel
                for extraction in ordered_chunk_extractions
                for rel in extraction.relationships
            ]
        )
        total_usage = REUsage(
            input_tokens=sum(e.usage.input_tokens for e in ordered_chunk_extractions),
            output_tokens=sum(e.usage.output_tokens for e in ordered_chunk_extractions),
            total_tokens=sum(e.usage.total_tokens for e in ordered_chunk_extractions),
            cost=sum(e.usage.cost for e in ordered_chunk_extractions),
        )

        return PipelineResult(
            entities=entities,
            relationships=all_relationships,
            chunk_extractions=ordered_chunk_extractions,
            usage=total_usage,
        )