# Contributing to de-bench

Keep contributions focused on tooling and documentation. Do not add benchmark collections, recorded model answers, third-party dictionaries or copied source documents. Use small synthetic fixtures for tests.

## User-supplied task schema

Pass your own JSON file or directory through `--tasks`. A file may contain one object or an array. The structural example below is a placeholder, not a benchmark question:

```json
{
  "id": "example-001",
  "domain": "hu",
  "category": "example",
  "difficulty": "medium",
  "prompt": "YOUR_QUESTION",
  "reference_answer": "YOUR_REFERENCE_ANSWER",
  "keywords": ["YOUR_EXPECTED_TERM"],
  "scoring": "keyword_presence",
  "source": "YOUR_SOURCE_AND_VERSION",
  "verified": false
}
```

The loader also accepts `reference` / `rubric`. Domain labels are `legal`, `automotive`, `hu`, `insurance` and `manufacturing`; difficulty labels are `easy`, `medium` and `hard`. IDs must be unique in the loaded selection. An explicit `verified: false` at the top level or in metadata excludes a task. A missing verification flag permits loading but does not establish source verification.

Keep task and output files outside the repository or in ignored local directories. Before running models, review your source rights, reference answers, required concepts, acceptable wording and disqualifying errors. Freeze the rubric and compare the same version across models. Changes to dictionary versions can change scores.

## Verification

```bash
python -m unittest discover -s tests
```

Validate loading, scoring and reporting offline with synthetic inputs. Do not make paid model calls in tests or commit credentials and private material.
