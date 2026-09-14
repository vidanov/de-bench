# de-bench

Tools for evaluating terminology coverage in German model answers. Supply your own questions, rubrics and optional language resources. This repository contains code and documentation only: no benchmark datasets, model answers, dictionaries or article results.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Inspect a task file you have prepared
de-bench list --tasks /absolute/path/to/tasks.json

# Evaluate a model already installed in your local Ollama service
de-bench run --tasks /absolute/path/to/tasks.json \
  -m "ollama/YOUR_INSTALLED_MODEL" -o results/local.json

# Compare runs of the same tasks
de-bench compare results/local.json results/other.json
```

`--tasks` accepts a JSON file or a directory searched recursively for JSON files. `-d hu`, `-d legal` and other supported domain labels optionally filter the supplied tasks. Files must contain task objects, not result logs. Explicitly unverified tasks are skipped and duplicate task IDs are rejected. See [the task schema](CONTRIBUTING.md).

For Bedrock, install `pip install -e '.[bedrock]'`, configure AWS credentials and use `-m "bedrock-converse/YOUR_MODEL_OR_PROFILE_ID@YOUR_AWS_REGION"`. Other adapters include `openai/`, `bedrock/`, `bedrock-mantle/` and `ollama-completion/`; see [runners.py](src/de_bench/runners.py). Replace placeholders with your actual configuration. OpenAI uses `OPENAI_API_KEY`; `.env` files are not loaded automatically.

## Scoring

`term_coverage` is the number of matched rubric terms divided by the number of expected terms. `avg_score` is the unweighted mean of per-task scores. Questions with different rubric sizes make these distinct statistics; the CLI reports both.

The keyword scorer uses case-insensitive substrings and heuristic German stem matching. Optional external dictionaries enable synonym expansion and compound decomposition; see [language resources](docs/language-resources.md). Missing dictionaries disable those layers. New runs record task snapshots, the submitted model specification, scorer hashes and dictionary hashes or their absence. Run outputs are local files; they can include the full input tasks and responses.

**Term coverage is not correctness.** A wrong sentence can contain the expected term, extra incorrect terms are not penalized, and permissive substring or partial-phrase matching can overcount. Missing synonyms can undercount. Review both hits and misses in context and evaluate correctness separately.

`exact_match` compares the complete stripped answer case-insensitively. `llm_judge` sends the task, reference and response to an external OpenAI model and requires separate credentials and validation. `human` requires manual annotation. The default keyword scorer makes no model calls beyond the model being evaluated.

Compare identical task wording, rubrics, resource versions and settings. The comparison command rejects different task IDs or recorded keyword rubrics, but that check alone does not establish identical prompts or configurations. Give repeated runs distinct output names; reusing a filename overwrites it.

## External references

These links point to the original sources; no source documents or datasets are copied into this repository.

- [OpenThesaurus](https://www.openthesaurus.de/about/download): optional synonym data.
- [german-decompounder](https://github.com/uschindler/german-decompounder): optional compound dictionary.
- [Gesetze im Internet](https://www.gesetze-im-internet.de/): official German statutory sources for your own task authoring.
- [EUR-Lex](https://eur-lex.europa.eu/): EU legal sources for your own task authoring.
- [AWS cross-Region inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html): routing documentation for deployment checks.

Use the terms and notices accompanying any external resource you obtain. The code license does not license those resources or establish rights in user-supplied tasks and model outputs.

## Development

```bash
python -m unittest discover -s tests
```

Tests use minimal synthetic fixtures, not domain benchmark datasets. Code lives in `src/de_bench/`. The optional `deep_analysis` module reads local `results/` and obtains tasks through `DE_BENCH_TASKS` (a file or directory), falling back to `./tasks`; it sends data to an external OpenAI judge.

Code and original documentation are MIT licensed; see [LICENSE](LICENSE).
