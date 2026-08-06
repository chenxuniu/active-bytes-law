# Public result artifacts

Only reviewed, non-identifying artifacts belong here:

- `manifests/`: campaign lock files and sanitized run manifests;
- `summaries/`: cell/repeat summaries used by the paper;
- `figures/`: generated paper figures with source-data references;
- `checksums/`: SHA-256 indexes for external archives.

`results/raw/`, `results/private/`, and `results/telemetry/` are ignored. Full
NCU/NSYS reports, raw logs, model weights, and unreviewed system output must not
be committed. Do not add fabricated placeholder results: an empty directory is
more honest than a synthetic result that can be mistaken for evidence.
