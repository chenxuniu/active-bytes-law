# Contributing

Changes to an unfrozen tool or document follow the usual pull-request workflow.
Changes to a frozen campaign are different: create a new campaign ID and state
the reason. Never silently overwrite a lock file or reuse a run ID.

Before opening a pull request:

```bash
python3 -m unittest discover -s tests -v
python3 scripts/check_publication_safety.py
```

Do not commit raw node inventories, unreviewed logs, credentials, model weights,
profiler binaries, or absolute paths. Synthetic test fixtures must use obviously
fictional identifiers. Report measurement exclusions and capacity-censored
cells rather than deleting them from the run index.
