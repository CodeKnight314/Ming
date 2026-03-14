from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Dict, List, Tuple

from tqdm import tqdm

from ming.core.prompts import REPORT_SECTION_WRITER_PROMPT
from ming.core.reference_cleanup import normalize_markdown_references
from ming.core.outline_parser import SectionPlan, SubsectionPlan, strip_markdown_fences
from ming.models.openrouter_model import OpenRouterModelConfig
from ming.subagent import Agent, AgentConfig, AgentResult
from ming.tools.kg_query_tool import KGQueryTool

logger = logging.getLogger(__name__)


@dataclass
class WriterAgentConfig:
    model: OpenRouterModelConfig
    kg_query_tool: KGQueryTool
    draft_output_path: str | None = None
    max_iterations: int = 16
    num_parallel_sections: int = 4


class WriterAgent:
    def __init__(self, config: WriterAgentConfig):
        self.config = config
        # We don't initialize a single self.agent because we'll create them per section for parallelization
        # to ensure state isolation if needed, or just reuse the logic.

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

    def _create_agent(self) -> Agent:
        # Increase max_tokens for full section generation
        model_config = self.config.model
        model_config.max_tokens = max(8192, model_config.max_tokens)

        return Agent(
            AgentConfig(
                model=model_config,
                system_prompt=REPORT_SECTION_WRITER_PROMPT,
                tools=[self.config.kg_query_tool],
                max_iterations=self.config.max_iterations,
            )
        )

    def _build_section_prompt(
        self,
        report_title: str,
        constraints_paragraph: str,
        section: SectionPlan,
    ) -> str:
        constraints = constraints_paragraph.strip() or "No additional constraints provided."

        subsection_guidelines = "\n".join([
            f"- {sub.title}: {sub.instruction}" for sub in section.subsections
        ])

        return (
            f"## TASK ASSIGNMENT\n"
            f"Report Title: {report_title}\n"
            f"Section to Write: {section.title}\n\n"
            f"### Section Context & Instructions\n"
            f"{section.instruction}\n\n"
            f"### Subsection Guidelines\n"
            f"{subsection_guidelines}\n\n"
            f"### Global Report Constraints\n"
            f"{constraints}\n\n"
            f"## INSTRUCTIONS\n"
            f"1. **Analyze**: Identify key entities and themes spanning all subsections of this section.\n"
            f"2. **Research**: Use `kg_query_tool` iteratively to gather facts, numbers, and causal links for the entire section.\n"
            f"3. **Target Length**: This entire section should be approximately 2000-2500 words.\n"
            f"4. **Citations**: Cite the URL using [URL] format from KG results.\n"
            f"5. **Write**: Output the full Markdown section starting with `## {section.title}` and including all `###` subsections."
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

    def _extract_urls_from_tool_results(self, messages: List[Dict[str, str]]) -> List[str]:
        urls = []
        url_pattern = re.compile(r"URL: (https?://\S+)")
        for msg in messages:
            if msg["role"] == "tool_result":
                found = url_pattern.findall(msg["content"])
                urls.extend(found)
        return list(set(urls))

    def _write_section(
        self,
        report_title: str,
        constraints_paragraph: str,
        section: SectionPlan,
    ) -> Tuple[int, str, List[str]]:
        agent = self._create_agent()

        prompt = self._build_section_prompt(
            report_title=report_title,
            constraints_paragraph=constraints_paragraph,
            section=section,
        )
        result = agent.run(prompt)

        # Extract URLs from tool history
        urls = self._extract_urls_from_tool_results(result.messages)

        markdown = self._clean_section_markdown(
            report_title=report_title,
            section_title=section.title,
            raw_text=result.output,
        )

        return section.section_id, markdown, urls
    def _process_citations(self, sections_text: List[str], all_urls: List[str]) -> Tuple[List[str], Dict[str, int]]:
        # Unique URLs sorted for deterministic numbering
        unique_urls = sorted(list(set(all_urls)))
        url_to_idx = {url: i + 1 for i, url in enumerate(unique_urls)}

        processed_sections = []
        for text in sections_text:
            processed_text = text
            
            # 1. Replace explicit [URL] citations
            for url, idx in url_to_idx.items():
                escaped_url = re.escape(url)
                # Matches [http://...] or [URL] 
                processed_text = re.sub(rf"\[{escaped_url}\]", f"[{idx}]", processed_text)
            
            # 2. Heuristically catch raw URLs that the model might have missed brackets for
            # but only if they are the exact URLs we found in the tool results
            for url, idx in url_to_idx.items():
                # Avoid replacing URLs inside the final References section or if already bracketed
                # This is a simple heuristic: replace only if preceded by a space or punctuation
                # and not already inside brackets.
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
    ) -> str:
        draft_path = self._resolve_draft_path(
            report_title=report_title,
            draft_output_path=draft_output_path or self.config.draft_output_path,
        )

        logger.info(f"Writing {len(sections)} sections in parallel (max_workers={self.config.num_parallel_sections})...")

        section_results = [None] * len(sections)
        all_found_urls = []

        with ThreadPoolExecutor(max_workers=self.config.num_parallel_sections) as executor:
            future_to_idx = {
                executor.submit(self._write_section, report_title, constraints_paragraph, section): i
                for i, section in enumerate(sections)
            }

            for future in tqdm(as_completed(future_to_idx), total=len(sections), desc="Writing Sections"):
                idx = future_to_idx[future]
                try:
                    section_id, markdown, urls = future.result()
                    section_results[idx] = markdown
                    all_found_urls.extend(urls)
                except Exception as e:
                    logger.error(f"Failed to write section {sections[idx].title}: {e}")
                    section_results[idx] = f"## {sections[idx].title}\n\n_Error writing this section._"

        # Filter out any None results
        valid_sections = [res for res in section_results if res is not None]

        # Process citations: Convert [URL] to [N] and build reference list
        processed_sections, citations = self._process_citations(valid_sections, all_found_urls)

        final_markdown = self._report_markdown(report_title, processed_sections, citations)
        final_markdown = normalize_markdown_references(final_markdown)
        draft_path.write_text(final_markdown, encoding="utf-8")

        return final_markdown