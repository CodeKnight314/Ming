from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit


REFERENCE_LINE_RE = re.compile(r"^\[(\d+)\]:\s*(\S.*?)\s*$")
INLINE_URL_CITATION_RE = re.compile(r"\[(https?://[^\]\s]+)\]")


def _clean_url(raw_url: str) -> str:
    url = raw_url.strip()

    # The writer can leak trailing quotes, brackets, and commas.
    while url and url[-1] in ".,;:')\"]":
        url = url[:-1].rstrip()

    return url


def canonicalize_url(url: str) -> str:
    parts = urlsplit(_clean_url(url))
    normalized_path = parts.path.rstrip("/")
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            normalized_path,
            parts.query,
            "",
        )
    )


def _extract_unique_urls(lines: list[str]) -> list[str]:
    unique_urls: list[str] = []
    seen_keys: set[str] = set()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        match = REFERENCE_LINE_RE.match(stripped)
        if not match:
            raise ValueError(f"Unsupported reference line: {line.rstrip()}")

        url = _clean_url(match.group(2))
        key = canonicalize_url(url)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_urls.append(url)

    return unique_urls


def _split_reference_section(text: str, heading: str) -> tuple[str | None, list[str], str]:
    marker = f"{heading}\n"
    heading_index = text.find(marker)
    if heading_index == -1:
        return None, [], text

    prefix = text[: heading_index + len(marker)]
    rest = text[heading_index + len(marker) :]
    lines = rest.splitlines()

    reference_lines: list[str] = []
    trailing_start = len(lines)

    for index, line in enumerate(lines):
        if not line.strip():
            if reference_lines:
                trailing_start = index
                break
            continue

        if REFERENCE_LINE_RE.match(line.strip()):
            reference_lines.append(line)
            continue

        trailing_start = index
        break

    trailing = "\n".join(lines[trailing_start:])
    return prefix, reference_lines, trailing


def normalize_markdown_references(markdown_text: str, heading: str = "## References") -> str:
    """Normalize inline URL citations and deduplicate numbered references."""
    prefix, reference_lines, trailing = _split_reference_section(markdown_text, heading)

    if prefix is None or not reference_lines:
        return markdown_text

    unique_urls = _extract_unique_urls(reference_lines)
    url_to_index = {
        canonicalize_url(url): index for index, url in enumerate(unique_urls, start=1)
    }

    def replace_inline_url(match: re.Match[str]) -> str:
        url = _clean_url(match.group(1))
        key = canonicalize_url(url)
        if key not in url_to_index:
            url_to_index[key] = len(unique_urls) + 1
            unique_urls.append(url)
        return f"[{url_to_index[key]}]"

    normalized_prefix = INLINE_URL_CITATION_RE.sub(replace_inline_url, prefix)
    normalized_references = [
        f"[{index}]: {url}" for index, url in enumerate(unique_urls, start=1)
    ]

    parts = [normalized_prefix.rstrip(), "", *normalized_references]
    if trailing:
        parts.extend(["", trailing])

    return "\n".join(parts).rstrip() + "\n"
