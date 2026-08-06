# Public data dictionary

## Campaign lock

| Field | Meaning |
|---|---|
| `campaign_id` | immutable campaign namespace |
| `source_sha256` | exact source configuration hash |
| `algorithm_version` | campaign expansion semantics |
| `cells` | unique cell definitions before repetition |
| `run_order` | deterministic randomized cyclic order |
| `lock_sha256` | canonical lock-content hash |

## Iteration trace

| Field | Unit / meaning |
|---|---|
| `run_id`, `episode_id`, `iteration_id` | immutable identifiers |
| `monotonic_start_ns`, `monotonic_end_ns` | engine-step boundaries |
| `active_request_ids` | requests scheduled in this step |
| `useful_tokens_by_request` | metered accepted useful tokens per request |
| `metered_useful_output_tokens` | sum of the preceding mapping |
| `runtime_seq_len_raw_by_request` | optional runtime-native length |
| `attended_length_by_request` | canonical historical length before KV write |
| `live_kv_blocks`, `allocated_kv_blocks` | runtime KV state |
| `accepted_tokens`, `rejected_tokens` | useful acceptance and rejection accounting |
| `speculative_draft_tokens` | drafted speculative tokens; zero in static baseline |
| `preemptions`, `swaps`, `recomputed_tokens` | scheduler exceptions |
| `prefix_cache_hits`, `offloaded_bytes` | forbidden baseline mechanisms |
| `scheduler_mode`, `attention_backend`, `graph_mode` | runtime contract |
| `kv_cache_dtype`, `weight_dtype` | byte treatments |

## Energy episode

| Field | Unit / meaning |
|---|---|
| `boundary` | must be `decode-only` for primary records |
| `sensor` | `DCGM_FI_DEV_TOTAL_ENERGY` or documented equivalent |
| `counter_start_mj`, `counter_end_mj` | cumulative millijoule endpoints |
| `go_monotonic_ns`, `decode_done_monotonic_ns` | metered interval |
| `counter_read_start_monotonic_ns`, `counter_read_end_monotonic_ns` | handshake reads |
| `metered_useful_tokens` | useful-token denominator |
| `decode_seconds` | marker-contained time |
| `integrated_power_joules` | optional independent telemetry integral |
| `qc_pass`, `qc_reasons` | machine-checkable disposition |

## Public manifest

The public device label is local to the study, for example `gpu-0`. Raw device
UUIDs, hostnames, network addresses, serials, user identities, sites, lease
metadata, and absolute paths are prohibited. `artifact_uri` is repository
relative or an approved public archive URL. Every artifact carries SHA-256 and
byte size.
