from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from threading import local
from typing import List, Union

from ming.extraction.ner_module import Entity, NERModule
from ming.extraction.re_module import REModule, RERunResult, REUsage, Relationship
from ming.models import OpenRouterModelConfig


@dataclass
class SentenceExtraction:
    sentence: str
    entities: List[Entity]
    relationships: List[Relationship]
    usage: REUsage


@dataclass
class PipelineResult:
    entities: List[Entity]
    relationships: List[Relationship]
    sentence_extractions: List[SentenceExtraction]
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

    def _group_entities_by_sentence(self, entities: List[Entity]) -> List[SentenceExtraction]:
        grouped: OrderedDict[str, List[Entity]] = OrderedDict()
        for entity in entities:
            grouped.setdefault(entity.sentence, []).append(entity)

        return [
            SentenceExtraction(
                sentence=sentence,
                entities=sentence_entities,
                relationships=[],
                usage=REUsage(),
            )
            for sentence, sentence_entities in grouped.items()
        ]

    def _extract_sentence_relationships(self, sentence_extraction: SentenceExtraction) -> SentenceExtraction:
        target_entities = list(OrderedDict.fromkeys(entity.text for entity in sentence_extraction.entities))
        re_result: RERunResult = self._get_re_module().run_with_metadata(
            sentence_extraction.sentence, target_entities
        )
        return SentenceExtraction(
            sentence=sentence_extraction.sentence,
            entities=sentence_extraction.entities,
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
        entities = self.ner.run(text)
        sentence_extractions = self._group_entities_by_sentence(entities)

        if not sentence_extractions:
            return PipelineResult(
                entities=[],
                relationships=[],
                sentence_extractions=[],
                usage=REUsage(),
            )

        max_workers = min(self.max_workers, len(sentence_extractions))
        results_by_sentence = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self._extract_sentence_relationships, sentence_extraction): sentence_extraction.sentence
                for sentence_extraction in sentence_extractions
            }
            for future in as_completed(future_map):
                sentence = future_map[future]
                results_by_sentence[sentence] = future.result()

        ordered_sentence_extractions = [
            results_by_sentence[sentence_extraction.sentence] for sentence_extraction in sentence_extractions
        ]
        all_relationships = self._dedupe_relationships(
            [
                relationship
                for sentence_extraction in ordered_sentence_extractions
                for relationship in sentence_extraction.relationships
            ]
        )
        total_usage = REUsage(
            input_tokens=sum(item.usage.input_tokens for item in ordered_sentence_extractions),
            output_tokens=sum(item.usage.output_tokens for item in ordered_sentence_extractions),
            total_tokens=sum(item.usage.total_tokens for item in ordered_sentence_extractions),
            cost=sum(item.usage.cost for item in ordered_sentence_extractions),
        )

        return PipelineResult(
            entities=entities,
            relationships=all_relationships,
            sentence_extractions=ordered_sentence_extractions,
            usage=total_usage,
        )

if __name__ == "__main__":
    import glob
    import json
    import os

    from tqdm import tqdm

    re_config = {
        "provider": "openrouter",
        "model_name": "qwen/qwen3.5-flash-02-23",
        "temperature": 0.0,
        "max_tokens": 256,
        "model_kwargs": {
            "reasoning": {"enabled": False},
        },
    }
    ner_model_name = "en_core_web_sm"
    max_workers = 4
    pipeline = NERREPipeline(re_config, ner_model_name, max_workers)

    file_paths = glob.glob("/Users/richardtang/Desktop/IC_Projects/Rae/artifacts/sino_soviet_relations_in_the_1960s_url_texts/*.txt")
    os.makedirs("results", exist_ok=True)
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    for file in tqdm(file_paths):
        with open(file, "r") as f:
            text = f.read()
            if len(text.split()) < 200:
                continue
            print(f"Processing {file}...")
            result = pipeline.run(text)
            total_cost += result.usage.cost
            total_input_tokens += result.usage.input_tokens
            total_output_tokens += result.usage.output_tokens
            with open(os.path.join("results", os.path.basename(file).replace(".txt", ".json")), "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(
                "File usage: "
                f"input_tokens={result.usage.input_tokens}, "
                f"output_tokens={result.usage.output_tokens}, "
                f"cost=${result.usage.cost:.6f}"
            )
            print(
                "Running total: "
                f"input_tokens={total_input_tokens}, "
                f"output_tokens={total_output_tokens}, "
                f"cost=${total_cost:.6f}"
            )
            print(f"Result saved to {os.path.join('results', os.path.basename(file).replace('.txt', '.json'))}")

        break
