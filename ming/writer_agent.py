from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from tqdm import tqdm
from ming.core.outline_parser import SectionPlan, SubsectionPlan, strip_markdown_fences
from ming.core.prompts import REPORT_SECTION_WRITER_PROMPT
from ming.models.openrouter_model import OpenRouterModelConfig
from ming.subagent import Agent, AgentConfig
from ming.tools.kg_query_tool import KGQueryTool


@dataclass
class WriterAgentConfig:
    model: OpenRouterModelConfig
    kg_query_tool: KGQueryTool
    draft_output_path: str | None = None
    max_iterations: int = 16


class WriterAgent:
    def __init__(self, config: WriterAgentConfig):
        self.config = config
        self.agent = Agent(
            AgentConfig(
                model=config.model,
                system_prompt=REPORT_SECTION_WRITER_PROMPT,
                tools=[config.kg_query_tool],
                max_iterations=config.max_iterations,
            )
        )

    def _default_draft_path(self, report_title: str) -> Path:
        slug = re.sub(r"[^a-z0-9]+", "_", report_title.lower()).strip("_")
        if not slug:
            slug = "report"
        return Path(f"{slug}_draft.md")

    def _resolve_draft_path(self, report_title: str, draft_output_path: str | None) -> Path:
        output_path = Path(draft_output_path) if draft_output_path else self._default_draft_path(report_title)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def _report_markdown(self, report_title: str, completed_sections: list[str]) -> str:
        parts = [f"# {report_title}"]
        if completed_sections:
            parts.append("\n\n".join(completed_sections))
        return "\n\n".join(parts).strip() + "\n"

    def _flush_draft(self, report_title: str, completed_sections: list[str], draft_path: Path) -> None:
        draft_path.write_text(
            self._report_markdown(report_title, completed_sections),
            encoding="utf-8",
        )

    def _build_subsection_prompt(
        self,
        report_title: str,
        constraints_paragraph: str,
        section: SectionPlan,
        subsection: SubsectionPlan,
        section_draft: str,
    ) -> str:
        constraints = constraints_paragraph.strip() or "No additional constraints provided."
        current_section_draft = section_draft.strip() or "_Initial section header draft._"
        return (
            f"## TASK ASSIGNMENT\n"
            f"Report Title: {report_title}\n"
            f"Current Section: {section.title}\n"
            f"Subsection to Write: {subsection.title}\n\n"
            f"### Section Context & Instructions\n"
            f"{section.instruction}\n\n"
            f"### Subsection Focus & Evidence Goals\n"
            f"{subsection.instruction}\n\n"
            f"### Global Report Constraints\n"
            f"{constraints}\n\n"
            f"### Current Section Draft (Context for Flow)\n"
            f"{current_section_draft}\n\n"
            f"## INSTRUCTIONS\n"
            f"1. **Analyze**: Identify key entities and gaps in the current draft that need evidence.\n"
            f"2. **Research**: Use `kg_query_tool` iteratively to gather specific facts, numbers, and causal links.\n"
            f"3. **Write**: After research, output the Markdown subsection starting with `### {subsection.title}`."
        )

    def _clean_subsection_markdown(
        self,
        report_title: str,
        section_title: str,
        subsection_title: str,
        raw_text: str,
    ) -> str:
        cleaned = strip_markdown_fences(raw_text).strip()
        lines = cleaned.splitlines()

        while lines and not lines[0].strip():
            lines.pop(0)

        removable_headings = {
            f"# {report_title}".strip().lower(),
            f"## {section_title}".strip().lower(),
            f"# {section_title}".strip().lower(),
        }
        while lines and lines[0].strip().lower() in removable_headings:
            lines.pop(0)
            while lines and not lines[0].strip():
                lines.pop(0)

        cleaned = "\n".join(lines).strip()
        heading_pattern = re.compile(
            rf"^###\s+{re.escape(subsection_title)}\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        if not heading_pattern.search(cleaned):
            cleaned = f"### {subsection_title}\n\n{cleaned}".strip()

        return cleaned.strip() + "\n"

    def _write_section(
        self,
        report_title: str,
        constraints_paragraph: str,
        section: SectionPlan,
    ) -> str:
        section_chunks = [f"## {section.title}"]

        for subsection in section.subsections:
            section_draft = "\n\n".join(section_chunks).strip()
            prompt = self._build_subsection_prompt(
                report_title=report_title,
                constraints_paragraph=constraints_paragraph,
                section=section,
                subsection=subsection,
                section_draft=section_draft,
            )
            subsection_markdown = self.agent.run(prompt)
            section_chunks.append(
                self._clean_subsection_markdown(
                    report_title=report_title,
                    section_title=section.title,
                    subsection_title=subsection.title,
                    raw_text=subsection_markdown,
                )
            )

        return "\n\n".join(chunk.strip() for chunk in section_chunks if chunk.strip()).strip()

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
        completed_sections: list[str] = []
        self._flush_draft(report_title, completed_sections, draft_path)

        for section in tqdm(sections):
            completed_sections.append(
                self._write_section(
                    report_title=report_title,
                    constraints_paragraph=constraints_paragraph,
                    section=section,
                )
            )
            self._flush_draft(report_title, completed_sections, draft_path)

        return self._report_markdown(report_title, completed_sections)