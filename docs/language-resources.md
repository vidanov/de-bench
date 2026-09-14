# Optional external language resources

The repository contains matching code only. It does not distribute or automatically download dictionaries.

Obtain resources directly from their original projects and follow the license accompanying the version you choose:

| Resource | Original source |
| --- | --- |
| German synonyms | [OpenThesaurus downloads](https://www.openthesaurus.de/about/download) |
| Compound components | [german-decompounder](https://github.com/uschindler/german-decompounder), `dictionary-de.txt` |

Store downloaded resources outside the checkout. The synonym reader expects UTF-8 text with one comma-separated synonym group per line. The compound reader expects one component per line, with optional `#` comments. Extract archives locally before configuring paths.

```bash
export DE_BENCH_SYNONYMS=/absolute/path/to/synonyms.txt
export DE_BENCH_DECOMPOUND=/absolute/path/to/dictionary-de.txt
de-bench run --tasks /absolute/path/to/tasks.json \
  -m "ollama/YOUR_INSTALLED_MODEL" -o results/local.json
```

Unset variables disable their respective matching layers. Configured unreadable paths fail before a model call. Each run records resource SHA-256 hashes, or null for absent resources. Resource contents are not embedded in run metadata.

Use identical resources when comparing models. Different versions or missing dictionaries can change coverage scores. The MIT license for this tooling does not override external resource licenses.
