"""de-bench CLI — run evaluations and compare results."""

import json
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .evaluators import evaluate
from .loader import load_tasks
from .models import BenchmarkRun, Domain
from .runners import get_runner
from .reporting import term_summary
from .synonyms import resource_hashes

console = Console()
RESULTS_DIR = Path("results")


@click.group()
def main():
    """de-bench: German-language LLM evaluation suite."""
    pass


@main.command()
@click.option("--model", "-m", required=True, help="Model spec: provider/model (e.g. openai/gpt-4o)")
@click.option("--tasks", "task_source", required=True, type=click.Path(exists=True),
              help="Your task JSON file or directory (loaded recursively).")
@click.option("--domain", "-d", type=click.Choice([d.value for d in Domain]), default=None)
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path")
def run(model: str, domain: str | None, output: str | None, task_source: str):
    """Run benchmark against a model."""
    domain_enum = Domain(domain) if domain else None
    try:
        tasks = load_tasks(domain_enum, source=task_source)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if not tasks:
        console.print("[red]No active tasks found in the supplied source.[/red]")
        raise SystemExit(1)

    console.print(f"[bold]de-bench[/bold] — Running {len(tasks)} tasks against [cyan]{model}[/cyan]")
    console.print()

    try:
        dictionaries = resource_hashes()
    except OSError as exc:
        raise click.ClickException(f"Cannot read configured dictionary: {exc}") from exc
    if any(value is None for value in dictionaries.values()):
        console.print("[yellow]External dictionaries are not fully configured; "
                      "only configured dictionary layers will be used.[/yellow]")
    runner = get_runner(model)
    benchmark = BenchmarkRun(
        model=runner.model_name,
        domain=domain_enum.value if domain_enum else None,
        timestamp=datetime.now(timezone.utc).isoformat(),
        metadata={
            "model_spec": model,
            "dictionary_sha256": dictionaries,
            "tasks": [asdict(task) for task in tasks],
            "scorer_sha256": {
                name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                for name in ("evaluators.py", "synonyms.py")
            },
        },
    )

    for i, task in enumerate(tasks, 1):
        console.print(f"  [{i}/{len(tasks)}] {task.id} ({task.category})...", end=" ")
        result = runner.run(task)
        result = evaluate(task, result)
        benchmark.results.append(result)

        color = "green" if result.score >= 0.8 else "yellow" if result.score >= 0.5 else "red"
        console.print(f"[{color}]{result.score:.2f}[/{color}] ({result.latency_ms}ms)")

    # Summary
    scores = [r.score for r in benchmark.results]
    benchmark.summary = {
        "total_tasks": len(scores),
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "perfect_scores": sum(1 for s in scores if s == 1.0),
        "total_tokens_in": sum(r.tokens_in for r in benchmark.results),
        "total_tokens_out": sum(r.tokens_out for r in benchmark.results),
        "total_latency_ms": sum(r.latency_ms for r in benchmark.results),
    }
    benchmark.summary.update(term_summary(benchmark.results))

    console.print()
    if benchmark.summary["expected_terms"]:
        console.print(f"[bold]Term coverage: {benchmark.summary['term_coverage']:.2%}[/bold] "
                      f"({benchmark.summary['matched_terms']}/{benchmark.summary['expected_terms']} terms)")
    console.print(f"[bold]Mean task score: {benchmark.summary['avg_score']:.2%}[/bold] "
                  f"({benchmark.summary['perfect_scores']}/{len(scores)} perfect)")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(output) if output else RESULTS_DIR / f"{model.replace('/', '_')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(benchmark.to_dict(), indent=2, ensure_ascii=False))
    console.print(f"Results saved to {out_path}")


@main.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True))
def compare(files: tuple[str, ...]):
    """Compare results from multiple benchmark runs."""
    if len(files) < 2:
        console.print("[red]Provide at least 2 result files to compare.[/red]")
        raise SystemExit(1)

    runs = []
    for f in files:
        data = json.loads(Path(f).read_text())
        runs.append(BenchmarkRun.from_dict(data))

    signatures = [
        sorted((r.task_id, tuple(sorted(r.details.get("hits", []) + r.details.get("misses", []))))
               for r in run.results)
        for run in runs
    ]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise click.ClickException("Runs contain different task IDs or rubric terms; compare the same suite.")
    for run in runs:
        run.summary.update(term_summary(run.results))

    table = Table(title="de-bench Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Term coverage", justify="right")
    table.add_column("Terms", justify="right")
    table.add_column("Mean task score", justify="right")
    table.add_column("Perfect", justify="right")
    table.add_column("Tasks", justify="right")
    table.add_column("Tokens (in/out)", justify="right")
    table.add_column("Total Latency", justify="right")

    for r in sorted(runs, key=lambda x: x.summary.get("term_coverage") or 0, reverse=True):
        s = r.summary
        table.add_row(
            r.model,
            f"{s['term_coverage']:.2%}" if s["term_coverage"] is not None else "n/a",
            f"{s['matched_terms']}/{s['expected_terms']}",
            f"{s.get('avg_score', 0):.2%}",
            str(s.get("perfect_scores", 0)),
            str(s.get("total_tasks", 0)),
            f"{s.get('total_tokens_in', 0)}/{s.get('total_tokens_out', 0)}",
            f"{s.get('total_latency_ms', 0) / 1000:.1f}s",
        )

    console.print(table)


@main.command(name="list")
@click.option("--tasks", "task_source", required=True, type=click.Path(exists=True),
              help="Your task JSON file or directory (loaded recursively).")
@click.option("--domain", "-d", type=click.Choice([d.value for d in Domain]), default=None)
def list_tasks(domain: str | None, task_source: str):
    """List available tasks."""
    domain_enum = Domain(domain) if domain else None
    try:
        tasks = load_tasks(domain_enum, source=task_source)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    table = Table(title=f"Tasks ({len(tasks)} total)")
    table.add_column("ID", style="cyan")
    table.add_column("Domain")
    table.add_column("Category")
    table.add_column("Difficulty")
    table.add_column("Scoring")

    for t in tasks:
        table.add_row(t.id, t.domain.value, t.category, t.difficulty.value, t.scoring.value)

    console.print(table)


if __name__ == "__main__":
    main()
