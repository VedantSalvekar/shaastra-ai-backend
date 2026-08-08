from app.evals.metrics import (
    EvalCase,
    keyword_coverage,
    load_cases,
    mean_scores,
    normalize_doc_id,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_case,
)
from app.schemas.langgraph_state import IntentType, RetrievalChunk

DOC_A = "https://example.ie/a/"
DOC_B = "https://example.ie/b/"
DOC_OTHER = "https://example.ie/other/"


def chunk(doc_id: str, score: float = 0.5, text: str = "text") -> RetrievalChunk:
    return RetrievalChunk(text=text, score=score, metadata={"doc_id": doc_id})


def case(*doc_ids: str) -> EvalCase:
    return EvalCase(id="c1", question="q", relevant_doc_ids=list(doc_ids))


def test_same_page_under_www_or_http_is_one_document():
    assert normalize_doc_id("https://www.rtb.ie/renting/") == normalize_doc_id(
        "http://rtb.ie/renting"
    )


def test_parent_url_does_not_count_as_a_match_for_its_child_page():
    chunks = [chunk("https://example.ie/visas/student-visas/")]
    assert precision_at_k(chunks, case("https://example.ie/visas/"), k=1) == 0.0


def test_precision_is_share_of_retrieved_chunks_that_are_relevant():
    chunks = [chunk(DOC_OTHER), chunk(DOC_A), chunk(DOC_A)]
    assert precision_at_k(chunks, case(DOC_A), k=3) == 2 / 3


def test_precision_of_empty_result_is_zero():
    assert precision_at_k([], case(DOC_A), k=5) == 0.0


def test_recall_is_share_of_labelled_documents_found():
    chunks = [chunk(DOC_A), chunk(DOC_OTHER)]
    assert recall_at_k(chunks, case(DOC_A, DOC_B), k=2) == 0.5


def test_recall_counts_each_document_once_even_with_repeated_chunks():
    chunks = [chunk(DOC_A), chunk(DOC_A)]
    assert recall_at_k(chunks, case(DOC_A, DOC_B), k=2) == 0.5


def test_recall_only_looks_at_the_top_k():
    chunks = [chunk(DOC_OTHER), chunk(DOC_A)]
    assert recall_at_k(chunks, case(DOC_A), k=1) == 0.0
    assert recall_at_k(chunks, case(DOC_A), k=2) == 1.0


def test_reciprocal_rank_reflects_position_of_first_relevant_chunk():
    assert reciprocal_rank([chunk(DOC_A)], case(DOC_A)) == 1.0
    assert reciprocal_rank([chunk(DOC_OTHER), chunk(DOC_A)], case(DOC_A)) == 0.5


def test_reciprocal_rank_is_zero_when_nothing_relevant_is_retrieved():
    assert reciprocal_rank([chunk(DOC_OTHER)], case(DOC_A)) == 0.0


def test_case_with_no_labels_scores_zero_instead_of_crashing():
    chunks = [chunk(DOC_A)]
    unlabelled = case()
    assert precision_at_k(chunks, unlabelled, k=1) == 0.0
    assert recall_at_k(chunks, unlabelled, k=1) == 0.0
    assert reciprocal_rank(chunks, unlabelled) == 0.0


def test_score_case_reports_the_rank_of_the_first_hit():
    chunks = [chunk(DOC_OTHER, 0.9), chunk(DOC_OTHER, 0.8), chunk(DOC_A, 0.7)]
    result = score_case(case(DOC_A), chunks, k=3)
    assert result.first_relevant_rank == 3
    assert round(result.reciprocal_rank, 4) == 0.3333
    assert result.top_score == 0.9


def test_mrr_is_the_mean_of_per_case_reciprocal_ranks():
    hit = score_case(case(DOC_A), [chunk(DOC_A)], k=1)
    miss = score_case(case(DOC_A), [chunk(DOC_OTHER)], k=1)
    means = mean_scores([hit, miss])
    assert means["cases"] == 2
    assert means["mrr"] == 0.5


def test_mean_scores_of_no_cases_is_zeroed():
    assert mean_scores([]) == {
        "cases": 0,
        "precision": 0.0,
        "recall": 0.0,
        "mrr": 0.0,
        "keyword_coverage": None,
    }


def test_intent_decides_which_collection_is_searched():
    legal = EvalCase(id="c", question="q", intent=IntentType.LEGAL_ONLY)
    mixed = EvalCase(id="c", question="q", intent=IntentType.MIXED)
    user = EvalCase(id="c", question="q", intent=IntentType.USER_ONLY)
    assert legal.collection == "legal-knowledge"
    assert mixed.collection == "legal-knowledge"
    assert user.collection == "user-documents"


def test_keyword_coverage_is_share_of_expected_facts_in_the_context():
    labelled = EvalCase(
        id="c", question="q", relevant_keywords=["20 hours", "term time"]
    )
    chunks = [chunk(DOC_A, text="You may work 20 hours per week")]
    assert keyword_coverage(chunks, labelled, k=5) == 0.5


def test_keyword_coverage_is_case_insensitive_and_spans_all_chunks():
    labelled = EvalCase(
        id="c", question="q", relevant_keywords=["20 Hours", "TERM TIME"]
    )
    chunks = [chunk(DOC_A, text="work 20 hours"), chunk(DOC_B, text="during term time")]
    assert keyword_coverage(chunks, labelled, k=5) == 1.0


def test_keyword_coverage_is_none_when_the_case_has_no_keywords():
    assert keyword_coverage([chunk(DOC_A)], case(DOC_A), k=5) is None


def test_keyword_coverage_never_affects_the_ranking_metrics():
    """Keywords are reported separately so they cannot inflate precision."""
    labelled = EvalCase(
        id="c",
        question="q",
        relevant_doc_ids=[DOC_A],
        relevant_keywords=["20 hours"],
    )
    chunks = [chunk(DOC_OTHER, text="You may work 20 hours per week")]
    assert keyword_coverage(chunks, labelled, k=5) == 1.0
    assert precision_at_k(chunks, labelled, k=5) == 0.0
    assert recall_at_k(chunks, labelled, k=5) == 0.0
    assert reciprocal_rank(chunks, labelled) == 0.0


def test_mean_keyword_coverage_ignores_cases_without_keyword_labels():
    with_keywords = EvalCase(id="a", question="q", relevant_keywords=["hit"])
    without = EvalCase(id="b", question="q", relevant_doc_ids=[DOC_A])
    scores = [
        score_case(with_keywords, [chunk(DOC_A, text="hit")], k=1),
        score_case(without, [chunk(DOC_A)], k=1),
    ]
    assert mean_scores(scores)["keyword_coverage"] == 1.0


def test_golden_dataset_loads_and_every_case_is_labelled():
    cases = load_cases()
    assert len(cases) >= 30
    assert len({c.id for c in cases}) == len(cases), "case ids must be unique"
    for case_ in cases:
        assert case_.question
        assert case_.relevant_doc_ids, f"{case_.id} has no relevant_doc_ids"
        assert case_.reference_answer, f"{case_.id} has no reference_answer"
        assert isinstance(case_.intent, IntentType)
