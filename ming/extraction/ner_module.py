import spacy
from typing import List
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
    }

    def __init__(self, model_name: str = "en_core_web_sm", chunk_sentence_limit: int = 5):
        self.nlp = spacy.load(model_name)
        self.chunk_sentence_limit = chunk_sentence_limit

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

    def is_valid_entity(self, text: str, label: str) -> bool:
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
        if "word count" in candidate.lower():
            return False

        letter_count = sum(ch.isalpha() for ch in candidate)
        if letter_count < 3:
            return False

        if candidate[0] in "-:;,.)]}" or candidate[-1] in "-:;,[({":
            return False

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
            if candidate in {"External", "References", "Footnotes"}:
                return False

        return True

    def _extract_entities_for_chunks(self, chunks: List[Chunk]) -> None:
        """Populate each chunk's entities list. Mutates chunks in place."""
        seen_global_spans = set()
        for chunk in chunks:
            chunk.entities = []
            doc = self.nlp(chunk.text)
            for ent in doc.ents:
                if ent.label_ not in self._ENTITY_OF_INTEREST:
                    continue
                if not self.is_valid_entity(ent.text, ent.label_):
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

    def split_text_into_chunks(self, text: str) -> List[Chunk]:
        doc = self.nlp(text)
        chunks = []
        chunk_sents = []
        chunk_start = None

        for sent in doc.sents:
            if chunk_start is None:
                chunk_start = sent.start_char
            chunk_sents.append(sent)
            if len(chunk_sents) >= self.chunk_sentence_limit:
                chunk_end = chunk_sents[-1].end_char
                chunk_text = text[chunk_start:chunk_end]
                chunks.append(
                    Chunk(text=chunk_text, start=chunk_start, end=chunk_end, entities=[])
                )
                chunk_sents = []
                chunk_start = None

        if chunk_sents:
            chunk_end = chunk_sents[-1].end_char
            chunk_text = text[chunk_start:chunk_end]
            chunks.append(
                Chunk(text=chunk_text, start=chunk_start, end=chunk_end, entities=[])
            )
        return chunks

    def run(self, text: str) -> List[Chunk]:
        text = self.preprocess_text(text)
        chunks = self.split_text_into_chunks(text)
        self._extract_entities_for_chunks(chunks)
        return chunks
