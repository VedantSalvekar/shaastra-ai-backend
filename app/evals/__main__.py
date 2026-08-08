"""Run the retrieval evaluation:  python3 -m app.evals

Searches the vector store with each golden-dataset question and reports
Precision@K, Recall@K, MRR and keyword coverage.
"""

import argparse
from pathlib import Path

from app.evals.metrics import CaseScore, EvalCase, load_cases, mean_scores, score_case
from app.services.vector_store import search_with_filters


def retrieve(case: EvalCase, k: int):
    kwargs = {"collection": case.collection, "query": case.question, "top_k": k}
    if case.user_id:
        kwargs["user_id"] = case.user_id
    return search_with_filters(**kwargs)


def _pct(value: float | None) -> str:
    return f"{value:>8.2f}" if value is not None else f"{'-':>8}"


def print_report(scores: list[CaseScore], k: int) -> None:
    means = mean_scores(scores)
    width = 86

    print()
    print("=" * width)
    print(f"RETRIEVAL EVALUATION   K={k}   cases={means['cases']}")
    print("=" * width)
    print(f"{'case':<30}{'P@K':>8}{'R@K':>8}{'RR':>8}{'rank':>7}{'keywords':>10}{'score':>9}")
    print("-" * width)

    for s in sorted(scores, key=lambda x: x.reciprocal_rank):
        rank = s.first_relevant_rank or "-"
        top = f"{s.top_score:.3f}" if s.top_score is not None else "-"
        coverage = f"{s.keyword_coverage:.2f}" if s.keyword_coverage is not None else "-"
        print(
            f"{s.case_id[:29]:<30}"
            f"{s.precision:>8.2f}{s.recall:>8.2f}{s.reciprocal_rank:>8.2f}"
            f"{str(rank):>7}{coverage:>10}{top:>9}"
        )

    print("-" * width)
    print(
        f"{'MEAN':<30}"
        f"{means['precision']:>8.2f}{means['recall']:>8.2f}{means['mrr']:>8.2f}"
        f"{'':>7}{_pct(means['keyword_coverage']):>10}"
    )
    print()
    print(f"Precision@{k}     : {means['precision']:.3f}  share of retrieved chunks that are relevant")
    print(f"Recall@{k}        : {means['recall']:.3f}  share of labelled documents found")
    print(f"MRR             : {means['mrr']:.3f}  1/rank of first relevant chunk, averaged")
    if means["keyword_coverage"] is not None:
        print(f"Keyword coverage: {means['keyword_coverage']:.3f}  share of expected facts present in the context")

    misses = [s.case_id for s in scores if s.reciprocal_rank == 0.0]
    if misses:
        print()
        print(f"Nothing relevant retrieved for {len(misses)} case(s): {', '.join(misses)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Shaastra retrieval quality")
    parser.add_argument("--k", type=int, default=5, help="how many chunks to retrieve")
    parser.add_argument("--dataset", type=Path, default=None, help="path to golden dataset")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N cases")
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    if args.limit:
        cases = cases[: args.limit]

    scores = [score_case(case, retrieve(case, args.k), args.k) for case in cases]
    print_report(scores, args.k)


if __name__ == "__main__":
    main()
