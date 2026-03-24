from __future__ import annotations
from dataclasses import dataclass
import html
import re
import xml.etree.ElementTree as ET


OUTLINE_BLOCK_RE = re.compile(
    r"<report_outline\b[\s\S]*?</report_outline>",
    re.IGNORECASE,
)
LINE_PREFIX_RE = re.compile(r"^\s*L\d+:", re.MULTILINE)


@dataclass(frozen=True)
class SubsectionPlan:
    subsection_id: str
    title: str
    description: str
    instruction: str


@dataclass(frozen=True)
class SectionPlan:
    section_id: str
    title: str
    depth_target: str
    instruction: str
    subsections: list[SubsectionPlan]


def strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned

    lines = cleaned.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_outline_block(text: str) -> str:
    cleaned = strip_markdown_fences(text)
    cleaned = LINE_PREFIX_RE.sub("", cleaned)
    match = OUTLINE_BLOCK_RE.search(cleaned)
    if not match:
        raise ValueError("Could not find a <report_outline>...</report_outline> block.")
    return match.group(0)


def escape_xml_text_nodes(xml_text: str) -> str:
    def _escape_text_segment(segment: str) -> str:
        return html.escape(html.unescape(segment), quote=False)

    parts: list[str] = []
    last_end = 0
    for match in re.finditer(r"<[^>]+>", xml_text):
        text_segment = xml_text[last_end:match.start()]
        if text_segment:
            parts.append(_escape_text_segment(text_segment))
        parts.append(match.group(0))
        last_end = match.end()

    trailing = xml_text[last_end:]
    if trailing:
        parts.append(_escape_text_segment(trailing))

    return "".join(parts)


def sanitize_outline_xml(outline_xml: str) -> str:
    return escape_xml_text_nodes(extract_outline_block(outline_xml))


def normalize_space(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def trim_terminal_punctuation(text: str) -> str:
    return text.rstrip(" .")


def paragraph_for_section(section_id: str, section_title: str, depth_target: str) -> str:
    clean_depth_target = trim_terminal_punctuation(depth_target)
    if clean_depth_target:
        return (
            f"Section {section_id}, \"{section_title}\", should be developed with the "
            f"following depth target: {clean_depth_target}."
        )
    return f"Section {section_id}, \"{section_title}\", should be developed as a major part of the report."


def paragraph_for_subsection(
    section_title: str,
    subsection_id: str,
    subsection_title: str,
    description: str,
) -> str:
    clean_description = trim_terminal_punctuation(description)

    sentence = (
        f"In subsection {subsection_id}, \"{subsection_title}\" within \"{section_title}\", "
        f"the report should {clean_description[0].lower() + clean_description[1:]}"
        if clean_description
        else f"In subsection {subsection_id}, \"{subsection_title}\" within \"{section_title}\", "
        f"the report should develop this topic."
    )
    return sentence + "."


def paragraph_for_constraint(index: int, text: str, rationale: str) -> str:
    clean_text = trim_terminal_punctuation(text)
    clean_rationale = trim_terminal_punctuation(rationale)
    paragraph = f"Constraint {index}: {clean_text}."
    if clean_rationale:
        paragraph += f" Rationale: {clean_rationale}."
    return paragraph


def constraints_to_paragraph(root: ET.Element) -> str:
    constraints = root.findall("./constraints/constraint")
    if not constraints:
        return ""

    constraint_paragraphs = [
        paragraph_for_constraint(
            index=index,
            text=normalize_space(constraint.findtext("text")),
            rationale=normalize_space(constraint.findtext("rationale")),
        )
        for index, constraint in enumerate(constraints, start=1)
    ]
    return "\n\n".join(constraint_paragraphs).strip()


def outline_to_sections(outline_xml: str) -> tuple[str, str, list[SectionPlan]]:
    root = ET.fromstring(sanitize_outline_xml(outline_xml))
    report_title = normalize_space(root.findtext("report_title"))
    constraints_paragraph = constraints_to_paragraph(root)
    sections: list[SectionPlan] = []

    for section in root.findall("./toc/section"):
        section_id = section.attrib.get("id", "unknown")
        section_title = normalize_space(section.findtext("title"))
        depth_target = normalize_space(section.findtext("depth_target"))
        section_instruction = paragraph_for_section(
            section_id=section_id,
            section_title=section_title,
            depth_target=depth_target,
        )
        subsection_plans: list[SubsectionPlan] = []
        for subsection in section.findall("./subsections/subsection"):
            subsection_id = subsection.attrib.get("id", "unknown")
            subsection_title = normalize_space(subsection.findtext("title"))
            description = normalize_space(subsection.findtext("description"))
            subsection_plans.append(
                SubsectionPlan(
                    subsection_id=subsection_id,
                    title=subsection_title,
                    description=description,
                    instruction=paragraph_for_subsection(
                        section_title=section_title,
                        subsection_id=subsection_id,
                        subsection_title=subsection_title,
                        description=description,
                    ),
                )
            )
        sections.append(
            SectionPlan(
                section_id=section_id,
                title=section_title,
                depth_target=depth_target,
                instruction=section_instruction,
                subsections=subsection_plans,
            )
        )

    return report_title, constraints_paragraph, sections


def outline_to_section_lists(outline_xml: str) -> tuple[str, str, list[list[str]]]:
    report_title, constraints_paragraph, sections = outline_to_sections(outline_xml)
    section_lists = [
        [section.instruction, *[subsection.instruction for subsection in section.subsections]]
        for section in sections
    ]
    return report_title, constraints_paragraph, section_lists


def outline_toc_summary(sections: list[SectionPlan], *, current_index: int | None = None) -> str:
    """Format the full outline as compact text for section writers (narrative awareness).

    When *current_index* is set, labels that section as **YOU ARE HERE** so the model
    knows its position relative to the rest of the report.
    """
    if not sections:
        return "(No sections in outline.)"

    lines: list[str] = []
    for i, sec in enumerate(sections):
        marker = ""
        if current_index is not None and i == current_index:
            marker = " **← YOU ARE HERE**"
        depth = sec.depth_target.strip()
        depth_line = f"    Depth target: {depth}" if depth else ""
        lines.append(f"{i + 1}. [{sec.section_id}] {sec.title}{marker}")
        if depth_line:
            lines.append(depth_line)
        for sub in sec.subsections:
            desc = sub.description.strip()
            desc_part = f" — {desc}" if desc else ""
            lines.append(f"   - {sub.subsection_id} {sub.title}{desc_part}")
        lines.append("")
    return "\n".join(lines).strip()
