#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure what a running ``vllm-srun`` server does over a time window, from its ``/metrics`` counters.

The runtime exports cumulative counters, so two samples a few seconds apart give the throughput the
server really delivered while your clients ran: episodes are not visible to the server, but requests,
questions, forwards, rows and tokens are. Nothing is sent to the model; it only issues ``GET /metrics``.

Reported per window:

* ``requests/s`` and ``questions/s`` answered (HTTP 200), and the count of other statuses (500 = the model
  produced a non-finite number or a forward raised; 429/503 = queue full or not ready).
* ``tokens/s`` over wall time and while a forward ran, ``forward_busy`` (share of the window with a forward
  on the GPU: near 100 % means more clients only queue), ``rows/forward`` and ``tokens/forward``.
* mean queue wait and the queue depth at the end of the window.

Usage:
    python benchmarks/server_window.py                          # one 20 s window on :8000
    python benchmarks/server_window.py --window 60 --repeat 5   # five 60 s windows
    python benchmarks/server_window.py --base-url http://localhost:8001 --json
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass

SAMPLE_LINE = re.compile(r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)$")
STATUS_LABEL = re.compile(r'status="(?P<status>\d+)"')

Metrics = dict[tuple[str, str], float]


@dataclass(frozen=True)
class WindowSummary:
    """Throughput of the server over one window."""

    window_seconds: float
    requests_ok_per_second: float
    questions_per_second: float
    failed_requests_by_status: dict[str, int]
    forwards_per_second: float
    rows_per_forward: float
    tokens_per_forward: float
    tokens_per_second_wall: float
    tokens_per_second_in_forward: float
    forward_busy_fraction: float
    mean_queue_wait_ms: float
    queue_depth: float

    def format(self) -> str:
        """Human-readable block for the terminal."""
        failed = ", ".join(f"{status}: {count}" for status, count in sorted(self.failed_requests_by_status.items()))
        return "\n".join(
            [
                f"window {self.window_seconds:.1f}s",
                f"  ok requests/s     {self.requests_ok_per_second:.2f}   failed: {failed or 'none'}",
                f"  questions/s       {self.questions_per_second:.2f}",
                f"  forwards/s        {self.forwards_per_second:.2f}   rows/forward {self.rows_per_forward:.2f}"
                f"   tokens/forward {self.tokens_per_forward:.0f}",
                f"  forward busy      {100 * self.forward_busy_fraction:.0f}% of wall time",
                f"  tokens/s          {self.tokens_per_second_wall:.0f} wall, "
                f"{self.tokens_per_second_in_forward:.0f} while in forward",
                f"  queue wait        {self.mean_queue_wait_ms:.0f} ms mean, depth {self.queue_depth:.0f} at the end",
            ]
        )


def parse_metrics(text: str) -> Metrics:
    """Parse Prometheus text into ``{(metric name, label string): value}``; comments and bad lines are skipped."""
    metrics: Metrics = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_LINE.match(line.strip())
        if match is None:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        metrics[(match.group("name"), match.group("labels") or "")] = value
    return metrics


def total(metrics: Metrics, name: str) -> float:
    """Sum of every label set of ``name`` (0 when the metric has not been exported yet)."""
    return sum(value for (metric, _), value in metrics.items() if metric == name)


def requests_by_status(metrics: Metrics) -> dict[str, float]:
    """Cumulative ``vllm_srun_requests_total`` per HTTP status."""
    counts: dict[str, float] = {}
    for (name, labels), value in metrics.items():
        if name != "vllm_srun_requests_total":
            continue
        match = STATUS_LABEL.search(labels)
        if match:
            counts[match.group("status")] = counts.get(match.group("status"), 0.0) + value
    return counts


def summarise(before: Metrics, after: Metrics, seconds: float) -> WindowSummary:
    """Throughput between two samples taken ``seconds`` apart."""
    seconds = max(seconds, 1e-9)

    def delta(name: str) -> float:
        return max(total(after, name) - total(before, name), 0.0)

    status_before, status_after = requests_by_status(before), requests_by_status(after)
    deltas = {status: status_after[status] - status_before.get(status, 0.0) for status in status_after}
    forwards = delta("vllm_srun_forward_duration_seconds_count")
    forward_seconds = delta("vllm_srun_forward_duration_seconds_sum")
    tokens = delta("vllm_srun_batch_tokens_sum")
    queued = delta("vllm_srun_queue_duration_seconds_count")
    return WindowSummary(
        window_seconds=seconds,
        requests_ok_per_second=deltas.get("200", 0.0) / seconds,
        questions_per_second=delta("vllm_srun_questions_total") / seconds,
        failed_requests_by_status={
            status: int(count) for status, count in deltas.items() if status != "200" and count > 0
        },
        forwards_per_second=forwards / seconds,
        rows_per_forward=delta("vllm_srun_batch_rows_sum") / forwards if forwards else 0.0,
        tokens_per_forward=tokens / forwards if forwards else 0.0,
        tokens_per_second_wall=tokens / seconds,
        tokens_per_second_in_forward=tokens / forward_seconds if forward_seconds else 0.0,
        forward_busy_fraction=min(forward_seconds / seconds, 1.0),
        mean_queue_wait_ms=1000 * delta("vllm_srun_queue_duration_seconds_sum") / queued if queued else 0.0,
        queue_depth=total(after, "vllm_srun_queue_depth"),
    )


def fetch_metrics(base_url: str, timeout: float = 10.0) -> Metrics:
    """``GET <base_url>/metrics`` parsed."""
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/metrics", timeout=timeout) as response:  # noqa: S310
        return parse_metrics(response.read().decode("utf-8"))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000", help="Server base URL (default: %(default)s).")
    parser.add_argument("--window", type=float, default=20.0, help="Seconds between the two samples (default: %(default)s).")
    parser.add_argument("--repeat", type=int, default=1, help="Number of consecutive windows (default: %(default)s).")
    parser.add_argument("--json", action="store_true", help="Print one JSON object per window instead of text.")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    if args.window <= 0 or args.repeat < 1:
        print("--window must be positive and --repeat at least 1", file=sys.stderr)
        return 2
    try:
        previous, previous_time = fetch_metrics(args.base_url), time.monotonic()
        for _ in range(args.repeat):
            time.sleep(args.window)
            current, current_time = fetch_metrics(args.base_url), time.monotonic()
            summary = summarise(previous, current, current_time - previous_time)
            print(json.dumps(asdict(summary)) if args.json else summary.format(), flush=True)
            previous, previous_time = current, current_time
    except OSError as error:
        print(f"cannot read metrics from {args.base_url}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
