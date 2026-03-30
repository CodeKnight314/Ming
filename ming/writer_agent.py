from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from ming.core.prompts import (
    CONCLUSION_SECTION_WRITER_PROMPT,
    INTRO_SECTION_WRITER_PROMPT,
    READABILITY_POLISH_PROMPT,
    REPORT_SECTION_WRITER_PROMPT,
    STITCH_TRANSITIONS_PROMPT,
)
from ming.core.reference_cleanup import canonicalize_url, normalize_markdown_references
from ming.core.outline_parser import SectionPlan, outline_toc_summary, strip_markdown_fences
from ming.models.openrouter_model import OpenRouterModel, OpenRouterModelConfig
from ming.subagent import Agent, AgentConfig
from ming.tools.kg_query_tool import KGQueryTool

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://[^\s\]\"'>,)]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


@dataclass
class WriterAgentConfig:
    model: OpenRouterModelConfig
    kg_query_tool: KGQueryTool
    fallback_model: OpenRouterModelConfig | None = None
    polish_model: OpenRouterModelConfig | None = None
    draft_output_path: str | None = None
    max_iterations: int = 18
    num_parallel_sections: int = 8
    enable_stitch_pass: bool = True


class WriterAgent:
    _INITIAL_EVIDENCE_LIMIT = 32
    _MIN_UNIQUE_URLS = 8
    _MAX_URL_SHARE = 0.35

    def __init__(self, config: WriterAgentConfig):
        self.config = config
        self.polish_model = OpenRouterModel(config.polish_model) if config.polish_model is not None else None
        # Lightweight single-shot calls (stitch pass) share the primary writer model config.
        self._stitch_llm = OpenRouterModel(config.model)

    @staticmethod
    def _slugify_report_title(report_title: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", report_title.strip().lower()).strip("-")
        return slug or "report"

    def _resolve_draft_path(
        self,
        report_title: str,
        draft_output_path: str | None,
    ) -> Path:
        if draft_output_path:
            path = Path(draft_output_path).expanduser()
        else:
            path = Path("outputs") / f"{self._slugify_report_title(report_title)}.md"

        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _report_markdown(
        report_title: str,
        processed_sections: List[str],
        citations: Dict[str, int],
    ) -> str:
        parts = [f"# {report_title}".strip()]

        if processed_sections:
            parts.append("\n\n".join(section.strip() for section in processed_sections if section.strip()))

        if citations:
            references = "\n".join(
                f"[{idx}]: {url}" for url, idx in sorted(citations.items(), key=lambda item: item[1])
            )
            parts.append(f"## References\n\n{references}")

        return "\n\n".join(part for part in parts if part.strip()).strip() + "\n"

    def _create_agent(
        self,
        model_config: OpenRouterModelConfig | None = None,
        *,
        chain_fallback: bool = True,
        system_prompt: str | None = None,
    ) -> Agent:
        primary = model_config or self.config.model
        fallback_eff = (
            self.config.fallback_model
            if chain_fallback and self.config.fallback_model is not None
            else None
        )
        sp = system_prompt if system_prompt is not None else REPORT_SECTION_WRITER_PROMPT

        return Agent(
            AgentConfig(
                model=primary,
                fallback_model=fallback_eff,
                system_prompt=sp,
                tools=[self.config.kg_query_tool],
                max_iterations=self.config.max_iterations,
                max_tool_calls_per_turn=8,
            )
        )

    def _run_writer_prompt_with_fallback(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Run the writer model; on failure retry with fallback model."""
        agent = self._create_agent(self.config.model, system_prompt=system_prompt)
        try:
            result = agent.run(prompt)
            return result.output, result.messages
        except Exception as exc:
            if not self.config.fallback_model:
                raise
            logger.warning(
                "Primary writer model failed; retrying with fallback model: %s",
                exc,
            )
            fallback_agent = self._create_agent(
                self.config.fallback_model,
                chain_fallback=False,
                system_prompt=system_prompt,
            )
            fallback_result = fallback_agent.run(prompt)
            return fallback_result.output, fallback_result.messages

    def _build_initial_query(self, section: SectionPlan) -> str:
        subsection_titles = ", ".join(subsection.title for subsection in section.subsections)
        return "\n".join(
            part
            for part in [
                f"Section title: {section.title}",
                f"Subsection titles: {subsection_titles}",
                f"Section instruction: {section.instruction}",
            ]
            if part.strip()
        )

    def _canonical_entities_context(
        self,
        limit: int = 80,
        *,
        section_entity_names: set[str] | None = None,
    ) -> str:
        """Build an enriched entity context string for the writer prompt.

        If *section_entity_names* is provided, only entities whose name (lowered)
        appears in the set are included — giving the writer a section-relevant
        subgraph instead of the full KG dump.
        """
        summaries = self.config.kg_query_tool.kg_store.get_enriched_entity_summaries(
            limit=limit,
            filter_entity_names=section_entity_names,
        )
        if not summaries:
            return "None available."

        lines: List[str] = []
        for s in summaries:
            preds = ", ".join(s["top_predicates"]) if s["top_predicates"] else "—"
            label = s["label"] or "?"
            lines.append(f"- {s['name']} ({label}, {s['rel_count']} facts): {preds}")
        return "\n".join(lines)

    @staticmethod
    def _extract_entity_names_from_cards(cards: List[Dict[str, Any]]) -> set[str]:
        """Extract subject/object entity names from evidence card fact strings.

        Fact format: ``subject -[predicate]-> object (object_type)``
        Returns a lowercased set of entity names for filtering.
        """
        _FACT_RE = re.compile(r"^(.+?)\s+-\[.*?\]->\s+(.+?)\s+\(.*\)$")
        names: set[str] = set()
        for card in cards:
            fact = card.get("fact", "")
            m = _FACT_RE.match(fact)
            if m:
                names.add(m.group(1).strip().lower())
                names.add(m.group(2).strip().lower())
        return names

    def _format_evidence_cards(
        self,
        evidence_result: Dict[str, Any],
        grouped_by_subsection: Dict[str, List[int]] | None = None,
    ) -> str:
        cards = evidence_result.get("cards") or []
        if not cards:
            return "No evidence cards were returned from the KG."

        def _format_card(index: int, card: Dict[str, Any]) -> List[str]:
            lines: List[str] = []
            lines.append(f"[Evidence Card {index}]")
            lines.append(f"Fact: {card.get('fact', '')}")
            supporting_urls = card.get("supporting_urls") or []
            if supporting_urls:
                lines.append("Supporting URLs: " + ", ".join(supporting_urls[:6]))
            for excerpt_index, chunk in enumerate(card.get("chunks") or [], start=1):
                if excerpt_index > 2:
                    break
                lines.append(f"Excerpt {excerpt_index} ({chunk.get('url', '')}): {chunk.get('excerpt', '')}")
            lines.append("")
            return lines

        if grouped_by_subsection:
            lines: List[str] = []
            emitted: set[int] = set()
            for sub_title, card_indices in grouped_by_subsection.items():
                lines.append(f"--- Evidence for: {sub_title} ---")
                for ci in card_indices:
                    if 0 <= ci < len(cards):
                        lines.extend(_format_card(ci + 1, cards[ci]))
                        emitted.add(ci)
            # Emit remaining cards under broad heading
            remaining = [i for i in range(len(cards)) if i not in emitted]
            if remaining:
                lines.append("--- Broad evidence ---")
                for ci in remaining:
                    lines.extend(_format_card(ci + 1, cards[ci]))
            return "\n".join(lines).strip()

        lines = []
        for index, card in enumerate(cards, start=1):
            lines.extend(_format_card(index, card))
        return "\n".join(lines).strip()

    def _build_audit_feedback(
        self,
        *,
        thin_pool: bool,
        cited_unique_urls: int,
        highest_url_share: float,
        unused_urls: List[str],
    ) -> str:
        instructions: List[str] = []
        if thin_pool:
            instructions.append(
                "Evidence in the KG is thin for this section. State that limitation plainly in the section prose instead of implying complete coverage."
            )
        if (
            cited_unique_urls < self._MIN_UNIQUE_URLS
            or highest_url_share > self._MAX_URL_SHARE
        ) and unused_urls:
            instructions.append(
                "Diversify citations across the available surfaced evidence. Use more distinct URLs and avoid letting one source dominate the section."
            )
            instructions.append(
                "Prefer these surfaced-but-unused URLs when they genuinely support the claim: "
                + ", ".join(unused_urls[:12])
            )
        return "\n".join(f"- {instruction}" for instruction in instructions)

    def _build_section_prompt(
        self,
        report_title: str,
        constraints_paragraph: str,
        section: SectionPlan,
        *,
        initial_evidence_text: str,
        canonical_entities_context: str,
        audit_feedback: str = "",
        user_query: str = "",
        report_structure_context: str = "",
        additional_context: str = "",
    ) -> str:
        constraints = constraints_paragraph.strip() or "No additional constraints provided."

        subsection_guidelines = "\n".join(
            [f"- {sub.title}: {sub.instruction}" for sub in section.subsections]
        )
        subsection_order = "\n".join(
            f"{idx}. {sub.title}" for idx, sub in enumerate(section.subsections, start=1)
        )
        audit_section = ""
        if audit_feedback.strip():
            audit_section = f"### Audit Feedback\n{audit_feedback.strip()}\n\n"

        user_query_section = ""
        if user_query.strip():
            user_query_section = (
                f"### Original User Query\n"
                f"This report was requested to answer the following query. Ensure your section contributes to addressing it:\n"
                f"{user_query.strip()}\n\n"
            )

        structure_block = ""
        if report_structure_context.strip():
            structure_block = (
                f"### Report Structure (full outline)\n"
                f"Use this to understand where this section sits in the narrative, what precedes/follows it, "
                f"and to align tone and hand-offs (without duplicating other sections' detailed content).\n\n"
                f"{report_structure_context.strip()}\n\n"
            )

        extra_block = ""
        if additional_context.strip():
            extra_block = f"{additional_context.strip()}\n\n"

        return (
            f"## TASK ASSIGNMENT\n"
            f"Report Title: {report_title}\n"
            f"Section to Write: {section.title}\n\n"
            f"{user_query_section}"
            f"{structure_block}"
            f"{extra_block}"
            f"### Section Context & Instructions\n"
            f"{section.instruction}\n\n"
            f"### Subsection Guidelines\n"
            f"{subsection_guidelines}\n\n"
            f"### Mandatory Subsection Order\n"
            f"{subsection_order}\n\n"
            f"### Global Report Constraints\n"
            f"{constraints}\n\n"
            f"### Canonical KG Entities\n"
            f"Canonical KG entities: {canonical_entities_context or 'None available.'}\n\n"
            f"### Initial KG Evidence Cards\n"
            f"{initial_evidence_text}\n\n"
            f"{audit_section}"
            f"## INSTRUCTIONS\n"
            f"1. **Analyze**: Identify key entities and themes spanning all subsections of this section.\n"
            f"2. **Research**: Start with the surfaced evidence cards above. If you need more detail, call `kg_query_tool` with `search_evidence` first for broader evidence and use `get_neighbors` or `find_connection` only for drill-down.\n"
            f"3. **Target Length**: This entire section should be approximately 2000-2500 words.\n"
            f"4. **Citations**: Cite source URLs using [URL] format from KG results. Cite multiple sources for one claim when corroboration or disagreement matters, using adjacent citations like [URL][URL].\n"
            f"5. **Order Discipline**: Follow the mandatory subsection order exactly. Do not discuss a later subsection in detail before its own `###` header.\n"
            f"6. **Scope Discipline**: Keep each subsection focused on its assigned topic. If evidence is more relevant later, save it for the later subsection.\n"
            f"7. **Depth over Breadth**: For each subsection, explain *why* things happen (causal mechanisms) not just *what* happened. Present trade-offs and tensions when they exist. Use comparison tables when comparing 3+ items or dimensions.\n"
            f"8. **Write**: Output the full Markdown section starting with `## {section.title}` and including all `###` subsections."
        )

    def _clean_section_markdown(
        self,
        report_title: str,
        section_title: str,
        raw_text: str,
    ) -> str:
        cleaned = strip_markdown_fences(raw_text).strip()
        lines = cleaned.splitlines()

        while lines and not lines[0].strip():
            lines.pop(0)

        removable_headings = {
            f"# {report_title}".strip().lower(),
        }
        while lines and lines[0].strip().lower() in removable_headings:
            lines.pop(0)
            while lines and not lines[0].strip():
                lines.pop(0)

        cleaned = "\n".join(lines).strip()

        heading_pattern = re.compile(
            rf"^##\s+{re.escape(section_title)}\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        if not heading_pattern.search(cleaned):
            cleaned = f"## {section_title}\n\n{cleaned}".strip()

        return cleaned.strip() + "\n"

    def _polish_report(self, markdown: str, user_query: str = "") -> str:
        """Run a readability polish pass over the assembled report.

        Adds an executive summary, smooths transitions, cleans prose, and
        normalizes citation formatting — without altering content or citations.
        Returns the original markdown unchanged if no polish_model is configured
        or if the model call fails.
        """
        if self.polish_model is None:
            return markdown
        prompt = READABILITY_POLISH_PROMPT.format(report=markdown, user_query=user_query)
        try:
            return self.polish_model.generate(prompt)
        except Exception:
            logger.exception("Readability polish pass failed; returning original report")
            return markdown

    def _extract_urls_from_tool_results(self, messages: List[Dict[str, str]]) -> List[str]:
        urls: List[str] = []
        for msg in messages:
            if msg["role"] == "tool_result":
                found = _URL_RE.findall(msg["content"])
                urls.extend(canonicalize_url(url) for url in found)
        return sorted(set(urls))

    def _extract_cited_url_counts(self, markdown: str) -> Counter[str]:
        counts: Counter[str] = Counter()
        for url in _URL_RE.findall(markdown):
            counts[canonicalize_url(url)] += 1
        return counts

    @staticmethod
    def _strip_top_level_heading_and_following_blank(body: str) -> str:
        """Remove the first `## ...` line and leading blank lines."""
        lines = body.splitlines()
        i = 0
        while i < len(lines) and not lines[i].strip().startswith("## "):
            i += 1
        if i < len(lines):
            i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        return "\n".join(lines[i:]).strip()

    @classmethod
    def _first_paragraph_excerpt(cls, markdown: str, *, max_sentences: int = 2) -> str:
        body = cls._strip_top_level_heading_and_following_blank(markdown)
        if not body:
            return ""
        paragraphs: List[str] = []
        buf: List[str] = []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                if buf:
                    paragraphs.append(" ".join(buf).strip())
                    buf = []
                if paragraphs:
                    break
            elif stripped.startswith("#"):
                # Skip subsection headers; keep scanning for first prose block.
                continue
            else:
                buf.append(stripped)
        if buf and not paragraphs:
            paragraphs.append(" ".join(buf).strip())
        text = paragraphs[0] if paragraphs else ""
        sents = [s for s in _SENT_SPLIT_RE.split(text) if s.strip()]
        return " ".join(sents[:max_sentences]).strip()

    @classmethod
    def _last_paragraph_excerpt(cls, markdown: str, *, max_sentences: int = 2) -> str:
        body = cls._strip_top_level_heading_and_following_blank(markdown)
        if not body:
            return ""
        paragraphs: List[str] = []
        buf: List[str] = []
        for line in body.splitlines():
            if not line.strip():
                if buf:
                    paragraphs.append(" ".join(buf).strip())
                    buf = []
            elif line.lstrip().startswith("### "):
                if buf:
                    paragraphs.append(" ".join(buf).strip())
                    buf = []
            elif line.lstrip().startswith("## "):
                if buf:
                    paragraphs.append(" ".join(buf).strip())
                    buf = []
            else:
                buf.append(line.strip())
        if buf:
            paragraphs.append(" ".join(buf).strip())
        text = paragraphs[-1] if paragraphs else ""
        sents = [s for s in _SENT_SPLIT_RE.split(text) if s.strip()]
        return " ".join(sents[-max_sentences:]).strip() if sents else ""

    def _search_initial_evidence(
        self, section: SectionPlan
    ) -> Tuple[Dict[str, Any], Dict[str, List[int]] | None]:
        """Search evidence per-subsection and merge into a single result.

        Returns (merged_evidence_dict, subsection_grouping) where the grouping
        maps subsection titles to card indices in the merged list.
        """
        if not section.subsections:
            result = self.config.kg_query_tool.search_evidence(
                query=self._build_initial_query(section),
                limit=self._INITIAL_EVIDENCE_LIMIT,
                diversify_by_url=True,
            )
            return result, None

        # One broad query + one per subsection
        broad_limit = 10
        per_sub_limit = max(4, (self._INITIAL_EVIDENCE_LIMIT - broad_limit) // len(section.subsections))

        broad_result = self.config.kg_query_tool.search_evidence(
            query=self._build_initial_query(section),
            limit=broad_limit,
            diversify_by_url=True,
        )

        sub_results: List[Tuple[str, Dict[str, Any]]] = []
        for sub in section.subsections:
            sub_query = f"{sub.title}: {sub.description}" if sub.description else sub.title
            sub_result = self.config.kg_query_tool.search_evidence(
                query=sub_query,
                limit=per_sub_limit,
                diversify_by_url=True,
            )
            sub_results.append((sub.title, sub_result))

        # Merge and deduplicate cards by fact string
        seen_facts: Dict[str, int] = {}  # fact -> index in merged list
        merged_cards: List[Dict[str, Any]] = []
        subsection_grouping: Dict[str, List[int]] = {}
        thin_pool_any = broad_result.get("thin_pool", False)

        # Add subsection cards first (higher priority for grouping)
        for sub_title, sub_result in sub_results:
            thin_pool_any = thin_pool_any or sub_result.get("thin_pool", False)
            indices: List[int] = []
            for card in sub_result.get("cards") or []:
                fact = card.get("fact", "")
                if fact in seen_facts:
                    indices.append(seen_facts[fact])
                    continue
                idx = len(merged_cards)
                seen_facts[fact] = idx
                merged_cards.append(card)
                indices.append(idx)
            subsection_grouping[sub_title] = indices

        # Add broad cards that weren't already seen
        for card in broad_result.get("cards") or []:
            fact = card.get("fact", "")
            if fact not in seen_facts:
                seen_facts[fact] = len(merged_cards)
                merged_cards.append(card)

        # Cap at budget
        merged_cards = merged_cards[: self._INITIAL_EVIDENCE_LIMIT]
        # Prune grouping indices that exceeded the cap
        for sub_title in subsection_grouping:
            subsection_grouping[sub_title] = [
                i for i in subsection_grouping[sub_title] if i < len(merged_cards)
            ]

        all_urls = {
            url
            for card in merged_cards
            for url in (card.get("supporting_urls") or [])
        }
        merged_result: Dict[str, Any] = {
            "query": self._build_initial_query(section),
            "limit": self._INITIAL_EVIDENCE_LIMIT,
            "thin_pool": thin_pool_any,
            "unique_url_count": len(all_urls),
            "cards": merged_cards,
        }
        return merged_result, subsection_grouping

    def _needs_audit_rerun(
        self,
        markdown: str,
        *,
        surfaced_urls: List[str],
        thin_pool: bool,
    ) -> tuple[bool, str]:
        citation_counts = self._extract_cited_url_counts(markdown)
        cited_unique_urls = len(citation_counts)
        total_citations = sum(citation_counts.values())
        highest_url_share = (
            max(citation_counts.values()) / total_citations
            if total_citations > 0
            else 1.0
        )
        unused_urls = sorted(set(surfaced_urls) - set(citation_counts.keys()))
        needs_diversification = (
            (
                cited_unique_urls < self._MIN_UNIQUE_URLS
                or highest_url_share > self._MAX_URL_SHARE
            )
            and bool(unused_urls)
        )
        needs_rerun = thin_pool or needs_diversification
        return needs_rerun, self._build_audit_feedback(
            thin_pool=thin_pool,
            cited_unique_urls=cited_unique_urls,
            highest_url_share=highest_url_share,
            unused_urls=unused_urls,
        )

    def _write_section(
        self,
        report_title: str,
        constraints_paragraph: str,
        section: SectionPlan,
        user_query: str = "",
        *,
        report_structure_context: str = "",
        additional_context: str = "",
        system_prompt: str | None = None,
    ) -> Tuple[str, str, List[str]]:
        initial_evidence, subsection_grouping = self._search_initial_evidence(section)
        initial_evidence_text = self._format_evidence_cards(
            initial_evidence, grouped_by_subsection=subsection_grouping
        )
        # Extract entity names from evidence cards to scope the entity context
        section_entity_names = self._extract_entity_names_from_cards(
            initial_evidence.get("cards") or []
        )
        canonical_entities_context = self._canonical_entities_context(
            section_entity_names=section_entity_names or None,
        )
        surfaced_urls = sorted(
            {
                canonicalize_url(url)
                for card in (initial_evidence.get("cards") or [])
                for url in (card.get("supporting_urls") or [])
            }
        )

        prompt = self._build_section_prompt(
            report_title=report_title,
            constraints_paragraph=constraints_paragraph,
            section=section,
            initial_evidence_text=initial_evidence_text,
            canonical_entities_context=canonical_entities_context,
            user_query=user_query,
            report_structure_context=report_structure_context,
            additional_context=additional_context,
        )
        output, messages = self._run_writer_prompt_with_fallback(
            prompt,
            system_prompt=system_prompt,
        )
        urls = sorted(set(surfaced_urls + self._extract_urls_from_tool_results(messages)))

        markdown = self._clean_section_markdown(
            report_title=report_title,
            section_title=section.title,
            raw_text=output,
        )

        needs_rerun, audit_feedback = self._needs_audit_rerun(
            markdown,
            surfaced_urls=surfaced_urls,
            thin_pool=bool(initial_evidence.get("thin_pool", False)),
        )
        if needs_rerun and audit_feedback.strip():
            rerun_prompt = self._build_section_prompt(
                report_title=report_title,
                constraints_paragraph=constraints_paragraph,
                section=section,
                initial_evidence_text=initial_evidence_text,
                canonical_entities_context=canonical_entities_context,
                audit_feedback=audit_feedback,
                user_query=user_query,
                report_structure_context=report_structure_context,
                additional_context=additional_context,
            )
            rerun_output, rerun_messages = self._run_writer_prompt_with_fallback(
                rerun_prompt,
                system_prompt=system_prompt,
            )
            urls = sorted(set(urls + self._extract_urls_from_tool_results(rerun_messages)))
            markdown = self._clean_section_markdown(
                report_title=report_title,
                section_title=section.title,
                raw_text=rerun_output,
            )

        return section.section_id, markdown, urls

    @staticmethod
    def _build_intro_body_excerpt_context(
        body_sections: list[SectionPlan],
        body_markdowns: list[str],
    ) -> str:
        lines = [
            "### Drafted body sections — opening excerpts",
            "Align the introduction with these specific themes (do not copy verbatim; foreshadow and frame).",
        ]
        for sec, md in zip(body_sections, body_markdowns):
            ex = WriterAgent._first_paragraph_excerpt(md)
            lines.append(f"- **{sec.title}**: {ex or '(no excerpt)'}")
        return "\n".join(lines)

    @staticmethod
    def _build_conclusion_body_excerpt_context(
        body_sections: list[SectionPlan],
        body_markdowns: list[str],
    ) -> str:
        lines = [
            "### Drafted body sections — closing excerpts",
            "Synthesize across these threads; do not contradict them.",
        ]
        for sec, md in zip(body_sections, body_markdowns):
            ex = WriterAgent._last_paragraph_excerpt(md)
            lines.append(f"- **{sec.title}**: {ex or '(no excerpt)'}")
        return "\n".join(lines)

    def _generate_transitions(
        self,
        ordered_markdowns: list[str],
        ordered_titles: list[str],
        user_query: str = "",
    ) -> list[str]:
        """Return one transition string per boundary (len == len(sections) - 1)."""
        n = len(ordered_markdowns)
        if n < 2:
            return []
        boundary_parts: list[str] = []
        for i in range(n - 1):
            tail = self._last_paragraph_excerpt(ordered_markdowns[i])
            head = self._first_paragraph_excerpt(ordered_markdowns[i + 1])
            boundary_parts.append(
                f"Boundary {i + 1}: after «{ordered_titles[i]}» → before «{ordered_titles[i + 1]}»\n"
                f"  End of earlier section: {tail or '(empty)'}\n"
                f"  Start of next section: {head or '(empty)'}\n"
            )
        boundaries_block = "\n".join(boundary_parts)
        prompt = STITCH_TRANSITIONS_PROMPT.format(boundaries_block=boundaries_block)
        if user_query.strip():
            prompt = (
                f"User request (context only; do not add new factual claims):\n{user_query.strip()}\n\n"
                + prompt
            )
        try:
            raw = self._stitch_llm.generate(prompt)
        except Exception:
            logger.exception("Stitch transitions generation failed; skipping bridges")
            return [""] * (n - 1)

        cleaned = strip_markdown_fences(raw.strip())
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Stitch transitions JSON parse failed; skipping bridges")
            return [""] * (n - 1)
        if not isinstance(data, list):
            return [""] * (n - 1)
        out = [str(x).strip() for x in data]
        need = n - 1
        if len(out) < need:
            out.extend([""] * (need - len(out)))
        return out[:need]

    @staticmethod
    def _interleave_sections_and_transitions(
        sections: list[str],
        transitions: list[str],
    ) -> list[str]:
        if not sections:
            return []
        out: list[str] = [sections[0]]
        for i in range(1, len(sections)):
            if i - 1 < len(transitions) and transitions[i - 1].strip():
                out.append(transitions[i - 1].strip())
            out.append(sections[i])
        return out

    def _process_citations(
        self,
        sections_text: List[str],
        all_urls: List[str],
    ) -> Tuple[List[str], Dict[str, int]]:
        canonical_to_url: Dict[str, str] = {}
        for url in all_urls:
            canonical_url = canonicalize_url(url)
            canonical_to_url.setdefault(canonical_url, canonical_url)

        unique_urls = sorted(canonical_to_url.values())
        url_to_idx = {url: i + 1 for i, url in enumerate(unique_urls)}

        processed_sections = []
        for text in sections_text:
            processed_text = text
            for url, idx in url_to_idx.items():
                escaped_url = re.escape(url)
                processed_text = re.sub(rf"\[{escaped_url}\]", f"[{idx}]", processed_text)
            for url, idx in url_to_idx.items():
                escaped_url = re.escape(url)
                processed_text = re.sub(rf"(?<!\[){escaped_url}(?!\])", f"[{idx}]", processed_text)

            processed_sections.append(processed_text)

        return processed_sections, url_to_idx

    def run(
        self,
        report_title: str,
        constraints_paragraph: str,
        sections: list[SectionPlan],
        draft_output_path: str | None = None,
        runtime_observer: Any = None,
        user_query: str = "",
    ) -> str:
        draft_path = self._resolve_draft_path(
            report_title=report_title,
            draft_output_path=draft_output_path or self.config.draft_output_path,
        )

        section_results: list[str | None] = [None] * len(sections)
        all_found_urls: list[str] = []

        use_three_phase = len(sections) >= 3
        progress_rows = [{"title": section.title, "status": "pending"} for section in sections]

        if use_three_phase:
            last_i = len(sections) - 1
            for i in range(len(sections)):
                if i == 0 or i == last_i:
                    progress_rows[i]["status"] = "pending"
                else:
                    progress_rows[i]["status"] = "running"
            logger.info(
                "Writing report: body sections in parallel, then intro/conclusion with body context "
                "(stitch=%s)...",
                self.config.enable_stitch_pass,
            )
        else:
            for row in progress_rows:
                row["status"] = "running"
            logger.info(
                f"Writing {len(sections)} sections in parallel "
                f"(max_workers={self.config.num_parallel_sections})..."
            )

        if runtime_observer is not None and sections:
            runtime_observer.update_run(
                metrics={
                    "write_sections_progress_json": json.dumps(
                        progress_rows, ensure_ascii=False
                    ),
                    "write_sections_completed": 0,
                    "write_sections_total": len(sections),
                }
            )

        def _emit_write_progress() -> None:
            if runtime_observer is None or not sections:
                return
            done = sum(1 for row in progress_rows if row["status"] == "done")
            runtime_observer.update_run(
                metrics={
                    "write_sections_progress_json": json.dumps(
                        progress_rows, ensure_ascii=False
                    ),
                    "write_sections_completed": done,
                    "write_sections_total": len(sections),
                }
            )

        def _record_result(idx: int, markdown: str, urls: List[str]) -> None:
            section_results[idx] = markdown
            all_found_urls.extend(urls)
            progress_rows[idx]["status"] = "done"
            _emit_write_progress()

        def _record_failure(idx: int, exc: Exception) -> None:
            logger.error(f"Failed to write section {sections[idx].title}: {exc}")
            section_results[idx] = f"## {sections[idx].title}\n\n_Error writing this section._"
            progress_rows[idx]["status"] = "failed"
            _emit_write_progress()

        def _write_at_index(idx: int, **kwargs: Any) -> Tuple[str, str, List[str]]:
            return self._write_section(
                report_title,
                constraints_paragraph,
                sections[idx],
                user_query,
                report_structure_context=outline_toc_summary(sections, current_index=idx),
                **kwargs,
            )

        if use_three_phase:
            body_indices = list(range(1, len(sections) - 1))
            with ThreadPoolExecutor(max_workers=self.config.num_parallel_sections) as executor:
                future_to_idx = {
                    executor.submit(_write_at_index, idx): idx for idx in body_indices
                }
                for future in tqdm(
                    as_completed(future_to_idx),
                    total=len(body_indices),
                    desc="Writing body sections",
                ):
                    idx_p = future_to_idx[future]
                    try:
                        _sid, md, urls = future.result()
                        _record_result(idx_p, md, urls)
                    except Exception as e:
                        _record_failure(idx_p, e)

            body_markdowns = [section_results[i] or "" for i in body_indices]
            body_plans = [sections[i] for i in body_indices]

            # Introduction (body-aware)
            intro_i = 0
            progress_rows[intro_i]["status"] = "running"
            _emit_write_progress()
            intro_ctx = self._build_intro_body_excerpt_context(body_plans, body_markdowns)
            try:
                _sid, md, urls = _write_at_index(
                    intro_i,
                    additional_context=intro_ctx,
                    system_prompt=INTRO_SECTION_WRITER_PROMPT,
                )
                _record_result(intro_i, md, urls)
            except Exception as e:
                _record_failure(intro_i, e)

            # Conclusion (body-aware)
            concl_i = len(sections) - 1
            progress_rows[concl_i]["status"] = "running"
            _emit_write_progress()
            concl_ctx = self._build_conclusion_body_excerpt_context(body_plans, body_markdowns)
            try:
                _sid, md, urls = _write_at_index(
                    concl_i,
                    additional_context=concl_ctx,
                    system_prompt=CONCLUSION_SECTION_WRITER_PROMPT,
                )
                _record_result(concl_i, md, urls)
            except Exception as e:
                _record_failure(concl_i, e)

        else:
            with ThreadPoolExecutor(max_workers=self.config.num_parallel_sections) as executor:
                future_to_idx = {
                    executor.submit(_write_at_index, i): i
                    for i in range(len(sections))
                }
                for future in tqdm(as_completed(future_to_idx), total=len(sections), desc="Writing Sections"):
                    idx = future_to_idx[future]
                    try:
                        _section_id, markdown, urls = future.result()
                        _record_result(idx, markdown, urls)
                    except Exception as e:
                        _record_failure(idx, e)

        # Filter out any None results (should not happen after failures handled)
        base_ordered = [res if res is not None else "" for res in section_results]
        titles_ordered = [s.title for s in sections]

        if (
            self.config.enable_stitch_pass
            and len(base_ordered) >= 2
            and any(base_ordered)
        ):
            transitions = self._generate_transitions(base_ordered, titles_ordered, user_query)
            valid_sections = self._interleave_sections_and_transitions(base_ordered, transitions)
        else:
            valid_sections = base_ordered

        # Process citations: Convert [URL] to [N] and build reference list
        processed_sections, citations = self._process_citations(valid_sections, all_found_urls)

        final_markdown = self._report_markdown(report_title, processed_sections, citations)
        final_markdown = normalize_markdown_references(final_markdown)
        final_markdown = self._polish_report(final_markdown, user_query=user_query)
        draft_path.write_text(final_markdown, encoding="utf-8")

        return final_markdown
