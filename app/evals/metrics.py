"""Golden dataset and retrieval quality metrics.

The dataset is the single source of ground truth; each field feeds a different
check:

    intent             -> which collection the question should be answered from
    relevant_doc_ids   -> Precision@K, Recall@K, MRR (document-level relevance)
    relevant_keywords  -> keyword coverage of the retrieved context
    reference_answer   -> ground truth for generation scoring (not used here)

Precision, recall and MRR all share one definition of relevance: a chunk is
relevant when it comes from a document listed in relevant_doc_ids. Keyword
coverage is reported separately so it can never distort the ranking metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.schemas.langgraph_state import IntentType, RetrievalChunk

GOLDEN_DATASET = Path(__file__).resolve().parents[2] / "evals" / "golden_dataset.json"


@dataclass
class EvalCase:
    id: str
    question: str
    intent: IntentType = IntentType.LEGAL_ONLY
    relevant_doc_ids: list[str] = field(default_factory=list)
    relevant_keywords: list[str] = field(default_factory=list)
    reference_answer: str = ""
    user_id: str | None = None

    @property
    def collection(self) -> str:
        """Questions about the user's own uploads are answered from their
        documents; everything else from the shared legal knowledge base."""
        if self.intent is IntentType.USER_ONLY:
            return "user-documents"
        return "legal-knowledge"


def load_cases(path: Path | None = None) -> list[EvalCase]:
    raw = json.loads((path or GOLDEN_DATASET).read_text(encoding="utf-8"))
    return [
        EvalCase(
            id=item["id"],
            question=item["question"],
            intent=IntentType(item.get("intent", IntentType.LEGAL_ONLY.value)),
            relevant_doc_ids=item.get("relevant_doc_ids", []),
            relevant_keywords=item.get("relevant_keywords", []),
            reference_answer=item.get("reference_answer", ""),
            user_id=item.get("user_id"),
        )
        for item in raw
    ]


def normalize_doc_id(value: str) -> str:
    """Canonicalise a document URL so the same page indexed under a different
    scheme, www prefix or trailing slash compares equal."""
    text = value.strip().lower()
    for scheme in ("https://", "http://"):
        if text.startswith(scheme):
            text = text[len(scheme) :]
            break
    if text.startswith("www."):
        text = text[len("www.") :]
    return text.rstrip("/")


def chunk_doc_id(chunk: RetrievalChunk) -> str | None:
    meta = chunk.metadata
    return meta.get("doc_id") or meta.get("source_url") or meta.get("url")


def matched_doc_id(chunk: RetrievalChunk, case: EvalCase) -> str | None:
    """The labelled document this chunk belongs to, or None if it is off-target."""
    doc_id = chunk_doc_id(chunk)
    if not doc_id:
        return None
    normalized = normalize_doc_id(doc_id)
    expected = {normalize_doc_id(d) for d in case.relevant_doc_ids}
    return normalized if normalized in expected else None


def is_relevant(chunk: RetrievalChunk, case: EvalCase) -> bool:
    return matched_doc_id(chunk, case) is not None


def precision_at_k(chunks: list[RetrievalChunk], case: EvalCase, k: int) -> float:
    """Of the chunks we handed to the LLM, what share was on-target?"""
    top = chunks[:k]
    if not top:
        return 0.0
    return sum(1 for c in top if is_relevant(c, case)) / len(top)


def recall_at_k(chunks: list[RetrievalChunk], case: EvalCase, k: int) -> float:
    """Of the documents that should have been found, what share appeared in top K?"""
    if not case.relevant_doc_ids:
        return 0.0
    found = {
        matched
        for chunk in chunks[:k]
        if (matched := matched_doc_id(chunk, case)) is not None
    }
    return len(found) / len({normalize_doc_id(d) for d in case.relevant_doc_ids})


def reciprocal_rank(chunks: list[RetrievalChunk], case: EvalCase) -> float:
    """1 / position of the first relevant chunk; 0 if none was retrieved."""
    for rank, chunk in enumerate(chunks, start=1):
        if is_relevant(chunk, case):
            return 1.0 / rank
    return 0.0


def keyword_coverage(
    chunks: list[RetrievalChunk], case: EvalCase, k: int
) -> float | None:
    """Share of the expected facts present in the retrieved text.

    Finding the right document is not the same as retrieving the chunk that
    actually states the answer, so this checks the context the LLM will read.
    Returns None when a case has no keyword labels.
    """
    if not case.relevant_keywords:
        return None
    context = " ".join(chunk.text for chunk in chunks[:k]).lower()
    hits = sum(1 for kw in case.relevant_keywords if kw.lower() in context)
    return hits / len(case.relevant_keywords)


@dataclass
class CaseScore:
    case_id: str
    question: str
    precision: float
    recall: float
    reciprocal_rank: float
    first_relevant_rank: int | None
    keyword_coverage: float | None
    top_score: float | None


def score_case(case: EvalCase, chunks: list[RetrievalChunk], k: int) -> CaseScore:
    rr = reciprocal_rank(chunks, case)
    return CaseScore(
        case_id=case.id,
        question=case.question,
        precision=precision_at_k(chunks, case, k),
        recall=recall_at_k(chunks, case, k),
        reciprocal_rank=rr,
        first_relevant_rank=round(1 / rr) if rr else None,
        keyword_coverage=keyword_coverage(chunks, case, k),
        top_score=chunks[0].score if chunks else None,
    )


def mean_scores(scores: list[CaseScore]) -> dict[str, float | None]:
    """Averages across cases. The mean of reciprocal ranks is the MRR."""
    if not scores:
        return {
            "cases": 0,
            "precision": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "keyword_coverage": None,
        }
    n = len(scores)
    covered = [s.keyword_coverage for s in scores if s.keyword_coverage is not None]
    return {
        "cases": n,
        "precision": sum(s.precision for s in scores) / n,
        "recall": sum(s.recall for s in scores) / n,
        "mrr": sum(s.reciprocal_rank for s in scores) / n,
        "keyword_coverage": sum(covered) / len(covered) if covered else None,
    }
