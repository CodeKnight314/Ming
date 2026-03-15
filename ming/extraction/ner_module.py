import spacy
from typing import List, Optional
from dataclasses import dataclass
import re

@dataclass
class Entity:
    text: str
    label: str
    global_start: int
    global_end: int


@dataclass
class Chunk:
    text: str
    start: int
    end: int
    entities: List[Entity]
    url: str
    embedding: Optional[List[float]] = None
    tfidf_embedding: Optional[List[float]] = None

class NERModule:
    _ENTITY_OF_INTEREST = {"ORG", "PERSON", "GPE", "PRODUCT", "EVENT", "FAC", "WORK_OF_ART", "LAW"}
    _SECTION_HEADERS_TO_TRUNCATE = {
        "see also",
        "footnotes",
        "references",
        "notes",
        "bibliography",
        "further reading",
        "external links",
        "参见",
        "参考资料",
        "参考文献",
        "注释",
        "延伸阅读",
        "外部链接",
    }
    _SPACY_SAFETY_LIMIT = 1_000_000
    _SPACY_SPLIT_BUFFER = 5_000
    _BOUNDARY_LOOKAHEAD = 2_000
    _BOUNDARY_LOOKBACK = 4_000
    _MIN_SEGMENT_CHARS = 20_000
    _PARAGRAPH_BOUNDARY_RE = re.compile(r"\n\s*\n")
    _SENTENCE_BOUNDARY_RE = re.compile(r"[.!?。！？；;](?:[\"'”’)]|）|】|》|」|』)?(?:\s+|$)")
    _WHITESPACE_BOUNDARY_RE = re.compile(r"\s+")

    def __init__(self, chunk_sentence_limit: int = 5):
        self.nlp_en = spacy.load("en_core_web_sm")
        self.nlp_zh = spacy.load("zh_core_web_sm")
        self.chunk_sentence_limit = chunk_sentence_limit

    def _is_chinese(self, text: str) -> bool:
        return any('\u4e00' <= char <= '\u9fff' for char in text)

    def clean_formula(self, text: str) -> str:
        formula_pattern = r'\\\[(.*?)\\\]'
        
        def process_formula(match):
            formula = match.group(1)

            formula = re.sub(r'\\quad\s*\([^)]*\)', '', formula)
            
            formula = formula.strip()
            
            return r'\[' + formula + r'\]'

        cleaned_text = re.sub(formula_pattern, process_formula, text)
        
        return cleaned_text
        
    def preprocess_text(self, text: str) -> str: 
        text = self.clean_formula(text)
        text = self.remove_non_content_sections(text)
        tab_pattern = re.compile(r'\t')
        multiple_newlines_pattern = re.compile(r'\n\s*\n')
        word_split_newline_pattern = re.compile(r'(\w)\n(\w)')
        multiple_spaces_pattern = re.compile(r' +')
        trailing_spaces_pattern = re.compile(r'\s+$')
        hyphen_pattern = re.compile(r'-\n')
        wiki_edit_pattern = re.compile(r'\[edit\]')

        text = re.sub(hyphen_pattern, '-', text)
        text = re.sub(wiki_edit_pattern, '', text)
        text = re.sub(tab_pattern, ' ', text)
        text = re.sub(multiple_newlines_pattern, '\n\n', text)
        text = re.sub(word_split_newline_pattern, r'\1 \2', text)
        text = re.sub(multiple_spaces_pattern, ' ', text)
        text = re.sub(trailing_spaces_pattern, '', text)
        return text

    def remove_non_content_sections(self, text: str) -> str:
        lines = text.splitlines()
        kept_lines = []
        for line in lines:
            normalized = line.strip().strip(":|- ").lower()
            if normalized in self._SECTION_HEADERS_TO_TRUNCATE:
                break
            if normalized.startswith("url:") or normalized.startswith("title:") or normalized.startswith("word count:"):
                continue
            if line.lstrip().startswith("|"):
                continue
            kept_lines.append(line)
        return "\n".join(kept_lines)

    def is_valid_entity(self, text: str, label: str, is_chinese: bool) -> bool:
        candidate = text.strip()
        if not candidate:
            return False

        invalid_substrings = ("[edit", "edit]", "{{", "}}", "http://", "https://")
        if any(fragment in candidate for fragment in invalid_substrings):
            return False
        if "\n" in candidate or "|" in candidate:
            return False
        if re.search(r"\[\d", candidate):
            return False
        if "word count" in candidate.lower() or "字数" in candidate:
            return False

        if is_chinese:
            # For Chinese, 2 characters is often enough (e.g. "张三", "华为")
            if len(candidate) < 2:
                return False
        else:
            letter_count = sum(ch.isalpha() for ch in candidate)
            if letter_count < 3:
                return False

        if candidate[0] in "-:;,.)]}" or candidate[-1] in "-:;,[({":
            return False

        if not is_chinese:
            if label == "PERSON":
                parts = candidate.split()
                if len(parts) < 2:
                    return False
                if re.fullmatch(r"[A-Z]\.", parts[-1]) or re.fullmatch(r"[A-Z][a-z]?\.", parts[-1]):
                    return False

            trailing_stopwords = {"and", "for", "from", "in", "of", "on", "the", "to", "with"}
            if candidate.split()[-1].lower() in trailing_stopwords:
                return False

        if label == "ORG":
            if candidate in {"External", "References", "Footnotes", "外部", "参考", "脚注"}:
                return False

        return True

    def _extract_entities_for_chunks(self, chunks: List[Chunk], is_chinese: bool) -> None:
        """Populate each chunk's entities list. Mutates chunks in place."""
        seen_global_spans = set()
        nlp = self.nlp_zh if is_chinese else self.nlp_en
        for chunk in chunks:
            chunk.entities = []
            doc = nlp(chunk.text)
            for ent in doc.ents:
                if ent.label_ not in self._ENTITY_OF_INTEREST:
                    continue
                if not self.is_valid_entity(ent.text, ent.label_, is_chinese):
                    continue
                global_start = chunk.start + ent.start_char
                global_end = chunk.start + ent.end_char
                span_key = (ent.text, ent.label_, global_start, global_end)
                if span_key in seen_global_spans:
                    continue
                seen_global_spans.add(span_key)
                chunk.entities.append(
                    Entity(
                        text=ent.text,
                        label=ent.label_,
                        global_start=global_start,
                        global_end=global_end,
                    )
                )

    def _find_split_boundary(self, text: str, start: int, ideal_end: int, hard_end: int) -> int:
        search_start = max(start, ideal_end - self._BOUNDARY_LOOKBACK)
        search_end = min(len(text), hard_end)
        window = text[search_start:search_end]

        min_acceptable = start + min(self._MIN_SEGMENT_CHARS, max(1, (ideal_end - start) // 2))

        for pattern in (
            self._PARAGRAPH_BOUNDARY_RE,
            self._SENTENCE_BOUNDARY_RE,
            self._WHITESPACE_BOUNDARY_RE,
        ):
            matches = list(pattern.finditer(window))
            for match in reversed(matches):
                boundary = search_start + match.end()
                if min_acceptable <= boundary <= hard_end:
                    return boundary

        return ideal_end

    def _split_text_for_spacy(self, text: str, nlp) -> List[tuple[str, int]]:
        hard_limit = max(
            self._MIN_SEGMENT_CHARS,
            min(getattr(nlp, "max_length", self._SPACY_SAFETY_LIMIT), self._SPACY_SAFETY_LIMIT)
            - self._SPACY_SPLIT_BUFFER,
        )
        if len(text) <= hard_limit:
            return [(text, 0)]

        segments: List[tuple[str, int]] = []
        start = 0
        while start < len(text):
            remaining = len(text) - start
            if remaining <= hard_limit:
                segments.append((text[start:], start))
                break

            ideal_end = start + hard_limit - self._BOUNDARY_LOOKAHEAD
            hard_end = min(len(text), start + hard_limit)
            split_at = self._find_split_boundary(text, start, ideal_end, hard_end)
            if split_at <= start:
                split_at = hard_end

            segments.append((text[start:split_at], start))
            start = split_at

        return segments

    def split_text_into_chunks(self, text: str, url: str, is_chinese: bool) -> List[Chunk]:
        nlp = self.nlp_zh if is_chinese else self.nlp_en
        chunks = []

        for segment_text, segment_offset in self._split_text_for_spacy(text, nlp):
            doc = nlp(segment_text)
            chunk_sents = []
            chunk_start = None

            for sent in doc.sents:
                if chunk_start is None:
                    chunk_start = sent.start_char
                chunk_sents.append(sent)
                if len(chunk_sents) >= self.chunk_sentence_limit:
                    chunk_end = chunk_sents[-1].end_char
                    chunk_text = segment_text[chunk_start:chunk_end]
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            start=segment_offset + chunk_start,
                            end=segment_offset + chunk_end,
                            entities=[],
                            url=url,
                        )
                    )
                    chunk_sents = []
                    chunk_start = None

            if chunk_sents:
                chunk_end = chunk_sents[-1].end_char
                chunk_text = segment_text[chunk_start:chunk_end]
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        start=segment_offset + chunk_start,
                        end=segment_offset + chunk_end,
                        entities=[],
                        url=url,
                    )
                )

        return chunks

    def run(self, text: str, url: str) -> List[Chunk]:
        is_chinese = self._is_chinese(text)
        text = self.preprocess_text(text)
        chunks = self.split_text_into_chunks(text, url, is_chinese)
        self._extract_entities_for_chunks(chunks, is_chinese)
        return chunks