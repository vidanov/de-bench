"""Scoring evaluators for task results."""

from .models import ScoringMethod, Task, TaskResult


def evaluate(task: Task, result: TaskResult) -> TaskResult:
    """Score a task result using the task's scoring method."""
    match task.scoring:
        case ScoringMethod.EXACT_MATCH:
            return _score_exact_match(task, result)
        case ScoringMethod.KEYWORD_PRESENCE:
            return _score_keyword_presence(task, result)
        case ScoringMethod.LLM_JUDGE:
            return _score_llm_judge(task, result)
        case ScoringMethod.HUMAN:
            result.score = 0.0
            result.details = {"note": "Requires human annotation"}
            return result


def _score_exact_match(task: Task, result: TaskResult) -> TaskResult:
    """Exact string match (case-insensitive, stripped)."""
    expected = task.reference.strip().lower()
    actual = result.response.strip().lower()
    result.score = 1.0 if expected == actual else 0.0
    result.details = {"match": result.score == 1.0}
    return result


def _score_keyword_presence(task: Task, result: TaskResult) -> TaskResult:
    """Check which rubric keywords are present in the response (with synonym expansion)."""
    from .synonyms import keyword_matches

    hits = []
    misses = []

    for keyword in task.rubric:
        if keyword_matches(keyword, result.response):
            hits.append(keyword)
        else:
            misses.append(keyword)

    result.score = len(hits) / len(task.rubric) if task.rubric else 0.0
    result.details = {"hits": hits, "misses": misses}
    return result


LLM_JUDGE_PROMPT = """Du bist ein strenger Evaluator für deutsche Fachtexte.

Bewerte die folgende Antwort anhand der Rubrik-Kriterien.
Vergib für jedes Kriterium 0 (nicht erfüllt) oder 1 (erfüllt).

## Aufgabe
{prompt}

## Referenzantwort
{reference}

## Zu bewertende Antwort
{response}

## Kriterien
{rubric}

Antworte NUR als JSON-Objekt mit dem Format:
{{"scores": {{"kriterium1": 0, "kriterium2": 1, ...}}, "reasoning": "kurze Begründung"}}
"""


def _score_llm_judge(task: Task, result: TaskResult) -> TaskResult:
    """Use GPT-4o as judge to score against rubric."""
    import json

    from openai import OpenAI

    client = OpenAI()
    rubric_str = "\n".join(f"- {r}" for r in task.rubric)

    judge_prompt = LLM_JUDGE_PROMPT.format(
        prompt=task.prompt,
        reference=task.reference,
        response=result.response,
        rubric=rubric_str,
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    try:
        judge_output = json.loads(resp.choices[0].message.content or "{}")
        scores = judge_output.get("scores", {})
        total = sum(scores.values())
        max_score = len(task.rubric)
        result.score = total / max_score if max_score > 0 else 0.0
        result.details = {
            "criterion_scores": scores,
            "reasoning": judge_output.get("reasoning", ""),
        }
    except (json.JSONDecodeError, AttributeError):
        result.score = 0.0
        result.details = {"error": "Failed to parse judge response"}

    return result
