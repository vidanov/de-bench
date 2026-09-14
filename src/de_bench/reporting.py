"""Pooled keyword coverage, distinct from the unweighted mean task score."""

from .models import TaskResult


def term_summary(results: list[TaskResult]) -> dict:
    keyword_results = [r for r in results if "hits" in r.details and "misses" in r.details]
    matched = sum(len(r.details["hits"]) for r in keyword_results)
    expected = matched + sum(len(r.details["misses"]) for r in keyword_results)
    return {
        "matched_terms": matched,
        "expected_terms": expected,
        "term_coverage": matched / expected if expected else None,
    }
