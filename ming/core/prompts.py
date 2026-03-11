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

ORCHESTRATOR_PLANNING_PROMPT = """
"""

ORCHESTRATOR_DECISION_PROMPT = """
"""

ORCHESTRATOR_VALIDATION_PROMPT = """
"""

REPORT_PROMPT = """
"""


def build_prompt(prompt: str, images: Optional[List[str]] = None) -> str:
    """Build a prompt for model generation. Returns the prompt string.
    For simple text-only prompts, use directly with model.generate(prompt, ...).
    When images are provided, they should be passed separately to model.generate.
    """
    return prompt
