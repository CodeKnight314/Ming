from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Entity:
    entity_id: str
    text: str
    label: str
    chunk_id: str
    relationships: List[str]
    resolved_id: str = "" # Reference to a CanonicalEntity


@dataclass
class CanonicalEntity:
    canonical_id: str
    text: str
    label: str
    entities: List[str]  # List of Entity IDs
    relationships: List[str] # Merged List of Relationship IDs


@dataclass
class Chunk:
    chunk_id: str
    text: str
    entities: List[str]
    url: str


@dataclass
class Relationship:
    relationship_id: str
    subject: str
    predicate: str
    object: str
    object_type: str
    confidence: float

