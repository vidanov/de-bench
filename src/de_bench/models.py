"""Core data models for de-bench."""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum


class Domain(str, Enum):
    LEGAL = "legal"
    AUTOMOTIVE = "automotive"
    HU = "hu"
    INSURANCE = "insurance"
    MANUFACTURING = "manufacturing"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ScoringMethod(str, Enum):
    EXACT_MATCH = "exact_match"
    KEYWORD_PRESENCE = "keyword_presence"
    LLM_JUDGE = "llm_judge"
    HUMAN = "human"


@dataclass
class Task:
    id: str
    domain: Domain
    category: str
    prompt: str
    reference: str
    scoring: ScoringMethod
    rubric: list[str] = field(default_factory=list)
    difficulty: Difficulty = Difficulty.MEDIUM
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        # Accept both schemas: reference/rubric (legal, automotive) and
        # reference_answer/keywords (hu, generator output).
        metadata = data.get("metadata", {})
        for key in ("source", "verified"):
            if key in data:
                metadata[key] = data[key]
        return cls(
            id=data["id"],
            domain=Domain(data["domain"]),
            category=data["category"],
            prompt=data["prompt"],
            reference=data.get("reference") or data.get("reference_answer", ""),
            scoring=ScoringMethod(data["scoring"]),
            rubric=data.get("rubric") or data.get("keywords", []),
            difficulty=Difficulty(data.get("difficulty", "medium")),
            metadata=metadata,
        )


@dataclass
class TaskResult:
    task_id: str
    model: str
    response: str
    score: float = 0.0
    details: dict = field(default_factory=dict)
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkRun:
    model: str
    timestamp: str
    domain: str | None = None
    results: list[TaskResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "timestamp": self.timestamp,
            "domain": self.domain,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkRun":
        results = [TaskResult(**r) for r in data.get("results", [])]
        return cls(
            model=data["model"],
            timestamp=data.get("timestamp", ""),
            domain=data.get("domain"),
            results=results,
            summary=data.get("summary", {}),
            metadata=data.get("metadata", {}),
        )
