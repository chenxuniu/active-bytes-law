# Publication safety

This repository is public. Generate public artifacts from an allowlist; do not
collect everything and assume that redaction will catch every identifier.

Never commit:

- account names, personal or enterprise email, credentials, SSH material, or
  tokens;
- hostnames, private addresses, BMC/OOB details, MAC addresses, device UUIDs,
  or serial numbers;
- lease, reservation, resource, health-request, rack, site, lab, pool, or
  internal topology metadata;
- private URLs, private Git remotes, absolute user paths, registry credentials,
  or unreviewed container environments;
- raw system dumps, complete logs, model weights, power streams, or profiler
  binaries.

The public system collector queries only OS, architecture, generic hardware
SKU/count/capacity, MIG and power policy, and software versions. It deliberately
does not query UUID, bus ID, serial, hostname, user, network, mount, or process
environment fields.

Before a commit:

```bash
python3 scripts/check_publication_safety.py
git diff --cached
```

Before a release, also scan Git history with an approved secret scanner and
review every compressed or binary artifact outside the normal workflow. If a
sensitive value has entered Git history, deleting the current file is not
enough: stop publication, clean the history, and rotate any credential.

Public release must comply with applicable institutional/employer open-source
review, model and dataset licenses, participant/privacy rules, and paper
artifact policy. This repository contains protocol and clean scaffolding only
until those checks and experiment QC pass.
