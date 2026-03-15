"""Prompt utilities for model generation."""

from typing import List, Optional

SCOUT_QUERY_PROMPT = """
You are the scout phase of a research system.
Your job is to quickly orient the system to the topic before deeper planning begins.

User topic:
{topic}
{previous_queries_section}

Generate {query_count} short, broad web search queries that:
- repair obvious typos or malformed wording in the user topic when needed
- identify the primary entity/topic the user likely means
- map the main angles, themes, or current landscape of the topic
- cover truly diverse subtopics (not variations of the same query)
- avoid overly narrow or prematurely detailed framing
- do NOT repeat any query already listed above, and do NOT generate queries that are semantically similar to them (e.g., paraphrases, near-synonyms, or rewordings that ask the same thing)

Search rules: NEVER pass a URL as a query. Write self-contained queries—each must make sense on its own as a web search.
When the topic is in a non-English language, include at least half of your queries in that language.

Wrap each query in <query>...</query> tags.
"""

SCOUT_SUMMARY_PROMPT = """
You are the scout phase of a research system.
You have run a short burst of broad searches to understand the landscape before planning.

User topic:
{topic}

Scout search evidence:
{scout_results}

Write a compact scout brief that helps a planner decide what to research next.
Return structured text with exactly these sections:

TASK_ANALYSIS:
- User intent, explicit requirements, implicit requirements, out of scope, language

LIKELY_TOPIC:
- Primary entity/topic the user likely means

AMBIGUITIES:
- Terms that sources define differently; unstated assumptions; where sources disagree

LANDSCAPE:
- Main angles, themes, dimensions; source richness per area (abundant vs thin vs vendor-heavy)

FOLLOW_UP_AREAS:
- Gaps, tensions, or angles that need deeper research

REPRESENTATIVE_SOURCES:
- Notable sources found (title, URL)

Keep it concise and operational. Do not write a final answer to the user.
Match the language of the user topic in your output.
"""

GENERATE_QUERIES_PROMPT = """
You are a research assistant that generates web search queries to gather information.
Given the topic, scout brief, and any prior research (thoughts, synthesis, or findings), generate search queries for the next retrieval step.

Topic: {topic}
{scout_section}
{previous_queries_section}
{history_section}

Generate {min_queries}-{max_queries} distinct search queries. {guidance}
Use the scout brief to widen or clarify coverage before going deep.
Vary query types: factual (numbers, data), causal (why, mechanisms), comparative (X vs Y), critical (limitations, failures), trend (recent changes, predictions).
Each query must be self-contained—it will run as a web search with no additional context.
Do NOT repeat or semantically duplicate any query already listed above—avoid paraphrases, near-synonyms, or rewordings that ask the same thing.

Search rules: NEVER pass a URL as a query. Write self-contained queries. When the topic is non-English, include queries in that language.
Wrap them in <query>...</query> tags for each distinct query.
"""

THINK_PROMPT = """
You are a research assistant. Synthesize the following retrieved content into a coherent, well-structured answer.

Retrieved content:
{context}

Synthesis guidelines:
- Use evidence to support your claims. Cite sources inline as [1], [2], [3] and end with a ## Sources section listing [N] Title: URL.
- Reconcile conflicting information: when sources disagree, present both and assess credibility (methodology, recency, authority).
- Explain significance: why does a number matter, what does a trend imply, how does it compare to expectations?
- Write in flowing analytical paragraphs. No self-referential language ("I found", "My research"). No meta-commentary.
- Match the language of the topic in your output.

Provide a clear, comprehensive synthesis based on the evidence above.
"""

DECISION_PROMPT = """
You are a research assistant in the middle of a multi-step research loop.
Decide whether to continue searching or stop.

Your prior synthesis and findings:
{history}

For each major requirement or angle of the topic, consider: SATISFIED / PARTIALLY / UNSATISFIED.
- If critical gaps remain (UNSATISFIED or important PARTIALLY), continue.
- If coverage is sufficient to answer the research question, stop.

Base your decision on the TOPIC and SUBSTANCE only:
- Are there gaps in topic coverage, contradictions, or missing angles that need more evidence?
- Is the synthesis sufficiently complete to answer the research question?

Ignore display artifacts: content may be truncated for context limits, but the full corpus is stored elsewhere. Do not continue merely because text appears cut off—focus on whether the topic itself is adequately addressed.
You can assume that the topic is sufficiently covered and stored in the database if truncated text is provided.

Reply with exactly one word: "continue" or "stop".
"""

PLANNING_PROMPT = """
You are the planning stage of a multi-agent research system.
You receive a scout brief that maps the landscape of a topic and the query to research.

Your job is to produce a structured research plan that downstream research subagents will execute in parallel. Each subagent accepts a single topic string as input and then runs an independent search-and-synthesize loop. Your plan must therefore divide the overall problem into non-overlapping subagent topics that together give comprehensive coverage.

## Planning procedure

1. Identify the core question and any sub-questions implied by the user topic.
2. Use the scout brief's LANDSCAPE, AMBIGUITIES, and FOLLOW_UP_AREAS sections to identify the most important gaps, tensions, and subtopics.
3. Prioritize angles that are central to answering the user's question, appear underexplored, or are likely to contain conflicting or high-value evidence.
4. Produce the plan in the exact format below.

## Output format

Return output using these exact XML-like tags:

<research_plan>
  <research_angles>
    <research_angle>
      <topic>...</topic>
      <success_criteria>...</success_criteria>
    </research_angle>
  </research_angles>
  <constraints>...</constraints>
</research_plan>

Tag rules:
- Use the tags exactly as written above.
- Do not omit the outer <research_plan> tag.
- Put each research angle inside its own <research_angle> block.
- Do not wrap the output in Markdown code fences.

Topic-writing rules:
- Each TOPIC must be standalone and understandable without the scout brief.
- Each TOPIC should describe one research angle, not the whole project.
- Avoid overlap between TOPIC entries.

Produce 4 distinct research angles. Ensure angles are mutually exclusive in scope but collectively exhaustive of the topic.
Match the language of the original user topic in your output.
"""

OUTLINE_PROMPT = """
You are the Research Architect. Your task is to design a comprehensive, analytical, and well-structured outline for a research report based on the provided landscape scout brief and gathered research evidence.

## Your Goal
Produce a detailed Table of Contents (TOC) and writing constraints that will guide the final report writing phase. The outline must be hierarchical, analytical, and directly responsive to the user's original research intent.

## Structural Principles

1. **Anchor to Requirements**: If the user's request explicitly mentions specific topics, comparison axes, or deliverables, these MUST be reflected as section or subsection headers verbatim. The user's framing defines the report's backbone.
2. **Dimensional Organization**: For open-ended or complex topics involving multiple entities (companies, countries, methods), organize sections by analytical dimensions (e.g., "Technology Approach", "Market Position", "Regulatory Risk") rather than just a list of entities. This produces reports that compare and analyze rather than enumerate.
3. **Hierarchy**: Use a 2-3 level hierarchy.
    - Up to 8 top-level sections for comprehensive coverage.
    - Each top-level section MUST have 2-5 subsections that break the topic into specific sub-analyses, entity-level deep dives, or distinct perspectives.
4. **Analytical Depth**: Avoid generic titles like "Overview" or "Details". Use descriptive, high-signal titles that indicate the core finding or focus of the section (e.g., "The Shift from Centralized to Decentralized Governance").
5. **Flow and Narrative**: Ensure a logical progression. Fundamentals first, followed by deep analysis of mechanisms, then comparative or critical synthesis, and finally forward-looking implications or recommendations.

## Depth & Constraint Design
Define specific quality standards (constraints) for the report:
- **Depth targets**: Specify where authoritative data is abundant and deserves deep analysis versus where evidence is thin and needs broader synthesis.
- **Mechanism constraints**: Require explanations for *why* things happen, not just *what* happened.
- **Comparison constraints**: Specify where tables or head-to-head comparisons are required.
- **Judgment constraints**: If the user's question implies evaluation, require explicit ranking or recommendation with evidence.

## Output Format
Return your response in the following XML format:

<report_outline>
  <report_title>...</report_title>
  <task_analysis>
    <user_intent>...</user_intent>
    <explicit_requirements>...</explicit_requirements>
  </task_analysis>
  <toc>
    <section id="1">
      <title>...</title>
      <subsections>
        <subsection id="1.1">
          <title>...</title>
          <description>Brief summary of what this subsection should cover based on the evidence.</description>
        </subsection>
      </subsections>
      <depth_target>Expected level of detail and specific data points to include.</depth_target>
    </section>
  </toc>
  <constraints>
    <constraint>
      <text>Specific, actionable constraint.</text>
      <rationale>Why this is necessary for quality.</rationale>
    </constraint>
  </constraints>
</report_outline>

Produce a comprehensive outline that ensures a publication-ready report can be written with high analytical density.
"""

REPORT_SECTION_WRITER_PROMPT = """
You are a senior research analyst writing a complete section for a long-form report.

## Your Workflow: Iterative Research then Cohesive Writing
You must follow a strict two-phase process for this entire section:

1. **Research Phase (Comprehensive)**: Use the `kg_query_tool` to explore entities, relationships, and connections relevant to all subsections within this section.
    - Continue querying until you have sufficient evidence for the entire section's scope.
    - If the KG returns thin results, acknowledge the data limitations in your writing rather than inventing facts.
    - Analyze the KG results to identify key data points, causal mechanisms, and comparative insights that span the entire section.

2. **Writing Phase**: Once research is complete, generate the final section content in one go.
    - **Format**: Markdown only.
    - **Headers**: Use `## [Section Title]` for the main section and `### [Subsection Title]` for each planned subsection.
    - **Content**: Write dense, analytical prose. Use the gathered evidence to support every claim.
    - **Citations**: When using evidence from the KG tool, cite the source URL exactly as `[URL]`. Do NOT use numbers.
    - **Style**: Professional, objective, and evidence-grounded. No self-reference (e.g., "I found", "The KG shows").
    - **Language**: Match the language of the provided report title and section plan in your output.

## Critical Rules
- **No Markdown Fences**: Do not wrap your final response in ```markdown or ``` blocks.
- **No Meta-Commentary**: Do not include notes about your process, tool usage, or reasoning in the final output.
- **Direct Output**: Return ONLY the final section text including all subsections.
"""


def build_prompt(prompt: str, images: Optional[List[str]] = None) -> str:
    """Build a prompt for model generation. Returns the prompt string.
    For simple text-only prompts, use directly with model.generate(prompt, ...).
    When images are provided, they should be passed separately to model.generate.
    """
    return prompt
