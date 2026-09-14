"""Load user-supplied JSON tasks; no benchmark data is bundled."""

import json
import os
from pathlib import Path

from .models import Domain, Task


def load_tasks(domain: Domain | None = None, source: str | Path | None = None) -> list[Task]:
    """Read a JSON file or a directory tree, optionally filtering by domain.

    Without an explicit source, use DE_BENCH_TASKS or ./tasks.
    """
    root = Path(source or os.environ.get("DE_BENCH_TASKS", "tasks")).expanduser()
    if not root.exists():
        if source is not None or os.environ.get("DE_BENCH_TASKS"):
            raise ValueError(f"Task source does not exist: {root}")
        return []
    files = [root] if root.is_file() else sorted(root.rglob("*.json"))
    tasks = []
    seen = set()
    for task_file in files:
        try:
            data = json.loads(task_file.read_text(encoding="utf-8"))
            entries = data if isinstance(data, list) else [data]
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError("Each task must be a JSON object")
                if entry.get("verified") is False or entry.get("metadata", {}).get("verified") is False:
                    continue
                task = Task.from_dict(entry)
                if domain is not None and task.domain != domain:
                    continue
                if task.id in seen:
                    raise ValueError(f"Duplicate task ID: {task.id}")
                seen.add(task.id)
                tasks.append(task)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid task file {task_file}: {exc}") from exc
    return tasks


def load_task_by_id(task_id: str, source: str | Path | None = None) -> Task | None:
    """Find a task in the user's task source."""
    return next((task for task in load_tasks(source=source) if task.id == task_id), None)
