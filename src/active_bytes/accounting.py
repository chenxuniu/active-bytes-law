"""Mechanism-aligned byte accounting for decode inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ActiveBytesComponents:
    """Logical active-byte terms per metered useful output token."""

    weight_bytes_per_token: float
    kv_read_bytes_per_token: float
    kv_write_bytes_per_token: float
    active_bytes_read: float
    active_bytes_read_write: float
    weight_kv_parity_ratio: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def kv_bytes_per_token(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    bytes_per_element: int,
    replication_factor: int = 1,
    padding_bytes_per_token: int = 0,
) -> int:
    """Return physical K+V storage bytes for one historical token.

    The factor of two represents key and value. ``replication_factor`` handles
    layouts that physically replicate KV state; padding is added after the
    replicated dense term.
    """

    values = {
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "bytes_per_element": bytes_per_element,
        "replication_factor": replication_factor,
    }
    for name, value in values.items():
        _positive_int(name, value)
    if isinstance(padding_bytes_per_token, bool) or not isinstance(
        padding_bytes_per_token, int
    ) or padding_bytes_per_token < 0:
        raise ValueError("padding_bytes_per_token must be a non-negative integer")
    dense = (
        2
        * n_layers
        * n_kv_heads
        * head_dim
        * bytes_per_element
        * replication_factor
    )
    return dense + padding_bytes_per_token


def active_weight_bytes(tensor_inventory: Iterable[Mapping[str, Any]]) -> int:
    """Deduplicate runtime weight storage and return resident physical bytes.

    Every inventory row must contain ``physical_storage_id`` and
    ``storage_nbytes``. Views and tied tensors may share an ID. Conflicting
    sizes for the same ID are rejected because silently picking one would make
    the accounting non-auditable.
    """

    storage: dict[str, int] = {}
    for index, row in enumerate(tensor_inventory):
        try:
            storage_id = str(row["physical_storage_id"])
            nbytes = row["storage_nbytes"]
        except KeyError as exc:
            raise ValueError(f"inventory row {index} is missing {exc.args[0]}") from exc
        if not storage_id:
            raise ValueError(f"inventory row {index} has an empty storage ID")
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
            raise ValueError(f"inventory row {index} has invalid storage_nbytes")
        previous = storage.get(storage_id)
        if previous is not None and previous != nbytes:
            raise ValueError(
                f"storage {storage_id!r} has inconsistent sizes: {previous} and {nbytes}"
            )
        storage[storage_id] = nbytes
    if not storage:
        raise ValueError("tensor inventory is empty")
    return sum(storage.values())


def summarize_trace(rows: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    """Compute realized B_eff and useful-token-weighted L_bar from trace rows."""

    iterations = 0
    useful_tokens = 0
    attended_positions = 0
    for index, row in enumerate(rows):
        iterations += 1
        try:
            per_request = row["useful_tokens_by_request"]
            attended = row["attended_length_by_request"]
            declared_total = row["metered_useful_output_tokens"]
        except KeyError as exc:
            raise ValueError(f"trace row {index} is missing {exc.args[0]}") from exc
        if not isinstance(per_request, Mapping) or not isinstance(attended, Mapping):
            raise ValueError(f"trace row {index} has non-object request accounting")
        row_total = 0
        for request_id, count in per_request.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"trace row {index} has invalid useful token count")
            if count == 0:
                continue
            if request_id not in attended:
                raise ValueError(
                    f"trace row {index} lacks attended length for {request_id!r}"
                )
            length = attended[request_id]
            if isinstance(length, bool) or not isinstance(length, (int, float)) or length < 0:
                raise ValueError(f"trace row {index} has invalid attended length")
            row_total += count
            attended_positions += count * length
        if row_total != declared_total:
            raise ValueError(
                f"trace row {index} declares {declared_total} useful tokens but maps {row_total}"
            )
        useful_tokens += row_total
    if iterations == 0:
        raise ValueError("trace is empty")
    if useful_tokens == 0:
        raise ValueError("trace contains no useful output tokens")
    return {
        "decode_iterations": iterations,
        "metered_useful_tokens": useful_tokens,
        "effective_batch": useful_tokens / iterations,
        "mean_attended_context": attended_positions / useful_tokens,
        "attended_historical_positions": attended_positions,
    }


def active_bytes(
    weight_bytes: int,
    kv_bytes_per_historical_token: int,
    effective_batch: float,
    mean_attended_context: float,
) -> ActiveBytesComponents:
    """Compute read-only and read+write Active-Bytes terms."""

    if isinstance(weight_bytes, bool) or not isinstance(weight_bytes, int) or weight_bytes <= 0:
        raise ValueError("weight_bytes must be a positive integer")
    if (
        isinstance(kv_bytes_per_historical_token, bool)
        or not isinstance(kv_bytes_per_historical_token, int)
        or kv_bytes_per_historical_token <= 0
    ):
        raise ValueError("kv_bytes_per_historical_token must be a positive integer")
    if not isinstance(effective_batch, (int, float)) or effective_batch <= 0:
        raise ValueError("effective_batch must be positive")
    if not isinstance(mean_attended_context, (int, float)) or mean_attended_context < 0:
        raise ValueError("mean_attended_context must be non-negative")

    weight_term = weight_bytes / effective_batch
    kv_read_term = kv_bytes_per_historical_token * mean_attended_context
    kv_write_term = float(kv_bytes_per_historical_token)
    read = weight_term + kv_read_term
    parity = kv_read_term / weight_term
    return ActiveBytesComponents(
        weight_bytes_per_token=weight_term,
        kv_read_bytes_per_token=kv_read_term,
        kv_write_bytes_per_token=kv_write_term,
        active_bytes_read=read,
        active_bytes_read_write=read + kv_write_term,
        weight_kv_parity_ratio=parity,
    )
