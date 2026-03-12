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

FINAL_REPORT_PROMPT = """
You are the final synthesis stage of a multi-agent research system.
You receive the accumulated research from multiple subagents—each covering a different angle of the topic—plus access to a knowledge graph containing structured facts extracted during the research.

Your job is to produce a single, authoritative, well-structured report that answers the original research question.

## Procedure

1. Read the accumulated research carefully, noting where subagents agree, disagree, or leave gaps.
2. Query the knowledge graph to:
   - Verify key claims against structured facts.
   - Discover connections between entities that individual subagents may not have linked.
   - Fill minor factual gaps without requiring another search cycle.
3. Write the report following the structure below.

## Report structure

### Title
<A clear, descriptive title for the report>

### Executive Summary
<3-5 sentence overview of the key findings and their significance>

### Background
<Context needed to understand the topic: definitions, historical context, scope of the question>

### Findings
Organize into thematic sections (not by subagent). Each section should:
- Present evidence from multiple subagents where available
- Cite sources inline as [1], [2], etc.
- Reconcile conflicting information: when sources disagree, present both sides and assess credibility based on methodology, recency, and authority
- Explain significance: why a number matters, what a trend implies, how a finding connects to the broader question

### Analysis
<Cross-cutting insights that emerge from combining the findings: patterns, causal chains, implications>

### Limitations & Open Questions
<What the research could not resolve, data gaps, areas where evidence is thin or contradictory>

### Conclusion
<Direct answer to the original research question, supported by the evidence above>

### Sources
<Numbered list: [N] Title — URL>

## Writing guidelines
- Write in flowing analytical paragraphs. No bullet-point dumps in the Findings or Analysis sections.
- No self-referential language ("I found", "Our research", "This report"). State findings directly.
- Integrate quantitative data (numbers, dates, statistics) where available—do not leave them buried in the source material.
- When a fact appears in the knowledge graph AND in subagent research, prefer the subagent version for context but cross-check the KG for accuracy.
- Match the language of the original user topic in your output.
"""


def build_prompt(prompt: str, images: Optional[List[str]] = None) -> str:
    """Build a prompt for model generation. Returns the prompt string.
    For simple text-only prompts, use directly with model.generate(prompt, ...).
    When images are provided, they should be passed separately to model.generate.
    """
    return prompt
