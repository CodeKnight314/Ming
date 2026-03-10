"""Prompt utilities for model generation."""

from typing import List, Optional

SCOUT_QUERY_PROMPT = """
You are the scout phase of a research system.
Your job is to quickly orient the system to the topic before deeper planning begins.

User topic:
{topic}

Generate {query_count} short, broad web search queries that:
- repair obvious typos or malformed wording in the user topic when needed
- identify the primary entity/topic the user likely means
- map the main angles, themes, or current landscape of the topic
- avoid overly narrow or prematurely detailed framing

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
LIKELY_TOPIC:
AMBIGUITIES:
LANDSCAPE:
FOLLOW_UP_AREAS:
REPRESENTATIVE_SOURCES:

Keep it concise and operational. Do not write a final answer to the user.
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
Wrap them in <query>...</query> tags for each distinct query.
"""

THINK_PROMPT = """
You are a research assistant. Synthesize the following retrieved content into a coherent, well-structured answer.
Use the evidence to support your claims and cite sources when relevant.

Retrieved content:
{context}

Provide a clear, comprehensive response based on the evidence above.
"""

DECISION_PROMPT = """
You are a research assistant in the middle of a multi-step research loop.
Decide whether to continue searching or stop.

Your prior synthesis and findings:
{history}

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
