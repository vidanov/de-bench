"""Deep analysis: LLM-as-judge comparison of model responses."""

import json
from pathlib import Path

from openai import OpenAI

RESULTS_DIR = Path("results")

JUDGE_PROMPT = """Du bist ein strenger Evaluator für deutsche Fachtexte. Vergleiche die folgenden Modellantworten auf dieselbe Fachfrage.

## Aufgabe
{prompt}

## Referenzantwort (Gold Standard)
{reference}

## Modellantworten

{responses_block}

## Bewertungskriterien (jeweils 0-5 Punkte)

1. **Fachliche Korrektheit** — Sind die Fakten richtig? Stimmen §§, Normen, technische Angaben?
2. **Vollständigkeit** — Werden alle wesentlichen Aspekte behandelt?
3. **Fachsprache** — Werden korrekte deutsche Fachbegriffe verwendet (nicht unnötige Anglizismen)?
4. **Struktur & Klarheit** — Ist die Antwort gut gegliedert und verständlich?
5. **Halluzinationsfreiheit** — Werden keine falschen §§, Normen oder Fakten erfunden?

## Ausgabeformat

Antworte NUR als JSON:
{{
  "rankings": [
    {{
      "model": "model_name",
      "scores": {{
        "fachliche_korrektheit": 0-5,
        "vollstaendigkeit": 0-5,
        "fachsprache": 0-5,
        "struktur_klarheit": 0-5,
        "halluzinationsfreiheit": 0-5
      }},
      "total": 0-25,
      "strengths": "kurz",
      "weaknesses": "kurz"
    }}
  ],
  "winner": "model_name",
  "analysis": "2-3 Sätze: Was unterscheidet die Antworten qualitativ?"
}}
"""


def load_results(path: Path) -> dict:
    return json.loads(path.read_text())


def run_deep_analysis(task_ids: list[str] | None = None):
    """Run LLM-as-judge on all collected results, comparing per task."""
    # Load all result files
    result_files = list(RESULTS_DIR.glob("*.json"))
    if not result_files:
        print("No results found.")
        return

    # Group results by task_id
    task_responses: dict[str, list[dict]] = {}
    task_meta: dict[str, dict] = {}

    for rf in result_files:
        data = load_results(rf)
        for r in data.get("results", []):
            tid = r["task_id"]
            if task_ids and tid not in task_ids:
                continue
            if tid not in task_responses:
                task_responses[tid] = []
            task_responses[tid].append({
                "model": data["model"],
                "response": r["response"],
                "keyword_score": r["score"],
            })

    # Load tasks for reference answers
    from de_bench.loader import load_tasks
    all_tasks = {t.id: t for t in load_tasks()}

    client = OpenAI()
    analyses = []

    for tid, responses in sorted(task_responses.items()):
        if len(responses) < 2:
            continue

        task = all_tasks.get(tid)
        if not task:
            continue

        print(f"\n{'='*60}")
        print(f"Analyzing: {tid} ({task.category})")
        print(f"{'='*60}")

        # Build comparison block
        responses_block = ""
        for i, r in enumerate(responses, 1):
            responses_block += f"### Modell: {r['model']} (keyword_score: {r['keyword_score']:.2f})\n"
            responses_block += f"{r['response'][:2000]}\n\n"

        prompt = JUDGE_PROMPT.format(
            prompt=task.prompt,
            reference=task.reference,
            responses_block=responses_block,
        )

        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        try:
            analysis = json.loads(resp.choices[0].message.content or "{}")
            analyses.append({"task_id": tid, "domain": task.domain.value, **analysis})

            # Print summary
            print(f"  Winner: {analysis.get('winner', '?')}")
            print(f"  Analysis: {analysis.get('analysis', '')}")
            for rank in analysis.get("rankings", []):
                print(f"    {rank['model']:40s} → {rank.get('total', '?')}/25  "
                      f"(+{rank.get('strengths', '')} / -{rank.get('weaknesses', '')})")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  ERROR: {e}")

    # Save full analysis
    output_path = RESULTS_DIR / "deep_analysis.json"
    output_path.write_text(json.dumps(analyses, indent=2, ensure_ascii=False))
    print(f"\n\nFull analysis saved to {output_path}")

    # Print summary leaderboard
    print("\n" + "=" * 60)
    print("AGGREGATE SCORES (LLM-as-Judge, /25 per task)")
    print("=" * 60)

    model_totals: dict[str, list[int]] = {}
    for a in analyses:
        for rank in a.get("rankings", []):
            model = rank["model"]
            if model not in model_totals:
                model_totals[model] = []
            model_totals[model].append(rank.get("total", 0))

    for model, scores in sorted(model_totals.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True):
        avg = sum(scores) / len(scores)
        print(f"  {model:45s} avg {avg:.1f}/25  (n={len(scores)})")


if __name__ == "__main__":
    run_deep_analysis()
