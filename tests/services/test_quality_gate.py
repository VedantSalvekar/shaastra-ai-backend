from app.schemas.langgraph_state import IntentType, RetrievalChunk
from app.services.quality_gate import validate_answer


def _legal_chunk(score: float = 0.4) -> RetrievalChunk:
    return RetrievalChunk(
        text="You must register your immigration permission and get an IRP.",
        score=score,
        metadata={"title": "Registration of non-EEA nationals", "source_url": "https://example.ie"},
    )


def _user_chunk(score: float = 0.4) -> RetrievalChunk:
    return RetrievalChunk(text="Your IRP shows Stamp 2.", score=score, metadata={"doc_id": "doc1"})


CITED_LEGAL_ANSWER = (
    "To register your permission you book an appointment and attend in person, "
    "bringing your passport and college letter. According to [Legal Source 1], "
    "you must register before your landing stamp expires."
)


def test_mixed_intent_keeps_strong_legal_answer_when_no_user_chunks():
    """Regression: a good cited legal answer must not be discarded for MIXED
    intent just because the user's documents had no relevant match."""
    is_valid, clarifying = validate_answer(
        draft_answer=CITED_LEGAL_ANSWER,
        legal_chunks=[_legal_chunk()],
        user_chunks=[],
        intent=IntentType.MIXED,
        user_has_docs=True,
    )
    assert is_valid is True
    assert clarifying is None


def test_mixed_intent_answers_legal_even_without_uploaded_docs():
    is_valid, clarifying = validate_answer(
        draft_answer=CITED_LEGAL_ANSWER,
        legal_chunks=[_legal_chunk()],
        user_chunks=[],
        intent=IntentType.MIXED,
        user_has_docs=False,
    )
    assert is_valid is True
    assert clarifying is None


def test_user_only_still_requires_documents():
    is_valid, clarifying = validate_answer(
        draft_answer="Your document shows Stamp 2. [User Document 1]",
        legal_chunks=[],
        user_chunks=[],
        intent=IntentType.USER_ONLY,
        user_has_docs=False,
    )
    assert is_valid is False
    assert clarifying is not None
    assert "upload" in clarifying.lower()


def test_user_only_with_docs_but_no_match_asks_clarification():
    is_valid, clarifying = validate_answer(
        draft_answer="Your document shows Stamp 2. [User Document 1]",
        legal_chunks=[],
        user_chunks=[],
        intent=IntentType.USER_ONLY,
        user_has_docs=True,
    )
    assert is_valid is False
    assert clarifying is not None


def test_mixed_without_legal_answer_falls_back_to_user_doc_prompt():
    """If the legal branch produced nothing usable, MIXED should still ask for docs."""
    is_valid, clarifying = validate_answer(
        draft_answer="I don't have enough information to answer.",
        legal_chunks=[],
        user_chunks=[],
        intent=IntentType.MIXED,
        user_has_docs=False,
    )
    assert is_valid is False
    assert clarifying is not None


def test_valid_mixed_answer_with_both_sources_passes():
    is_valid, clarifying = validate_answer(
        draft_answer=CITED_LEGAL_ANSWER + " Your IRP confirms this [User Document 1].",
        legal_chunks=[_legal_chunk()],
        user_chunks=[_user_chunk()],
        intent=IntentType.MIXED,
        user_has_docs=True,
    )
    assert is_valid is True
    assert clarifying is None
