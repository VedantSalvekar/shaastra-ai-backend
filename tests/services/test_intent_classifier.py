from app.schemas.langgraph_state import IntentType
from app.services.intent_classifier import (
    classify_intent,
    references_uploaded_document,
)

# General legal questions phrased with "I"/"my" must NOT be treated as
# document references — this was the root cause of legal answers being
# discarded by the quality gate.
LEGAL_QUESTIONS = [
    "How do I register my immigration permission after arriving in Ireland?",
    "How long is the landing stamp valid on a student visa?",
    "How many hours can I work on Stamp 2?",
    "Do I need private health insurance for my student visa?",
    "What documents do I need to open a bank account?",
    "Am I allowed to work part-time as a student?",
    "Do I have to register with immigration?",
]

DOC_QUESTIONS = [
    "What does my document say about my stamp?",
    "Read my cover letter and tell me the employer.",
    "What is in my uploaded document?",
    "Check my document for the expiry date.",
]


def test_general_legal_questions_are_not_document_references():
    for q in LEGAL_QUESTIONS:
        assert references_uploaded_document(q) is False, q


def test_explicit_document_questions_are_detected():
    for q in DOC_QUESTIONS:
        assert references_uploaded_document(q) is True, q


def test_document_reference_without_upload_returns_user_only():
    # Deterministic path (no LLM call): explicit doc reference + no uploads.
    result = classify_intent(
        "What does my document say?",
        user_has_docs=False,
    )
    assert result == IntentType.USER_ONLY
