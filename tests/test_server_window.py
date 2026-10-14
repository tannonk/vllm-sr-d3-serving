#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for benchmarks/server_window.py (no server needed).

Covers the Prometheus text parsing (labels, comments, junk lines) and the window arithmetic: rates,
rows and tokens per forward, forward-busy share, failed requests per status, and the empty-window case.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "benchmarks" / "server_window.py"

BEFORE = """\
# HELP vllm_srun_requests_total Requests by endpoint and status.
# TYPE vllm_srun_requests_total counter
vllm_srun_requests_total{endpoint="/v1/systemone",status="200"} 100.0
vllm_srun_requests_total{endpoint="/v1/systemone",status="500"} 2.0
vllm_srun_questions_total{outcome="answered",type="choice"} 190.0
vllm_srun_forward_duration_seconds_count 120.0
vllm_srun_forward_duration_seconds_sum 40.0
vllm_srun_batch_rows_sum 200.0
vllm_srun_batch_tokens_sum 120000.0
vllm_srun_queue_duration_seconds_count 100.0
vllm_srun_queue_duration_seconds_sum 0.5
vllm_srun_queue_depth 0.0
"""

AFTER = """\
vllm_srun_requests_total{endpoint="/v1/systemone",status="200"} 130.0
vllm_srun_requests_total{endpoint="/v1/systemone",status="500"} 5.0
vllm_srun_requests_total{endpoint="/v1/systemone",status="429"} 1.0
vllm_srun_questions_total{outcome="answered",type="choice"} 250.0
vllm_srun_forward_duration_seconds_count 160.0
vllm_srun_forward_duration_seconds_sum 50.0
vllm_srun_batch_rows_sum 260.0
vllm_srun_batch_tokens_sum 150000.0
vllm_srun_queue_duration_seconds_count 134.0
vllm_srun_queue_duration_seconds_sum 8.7
vllm_srun_queue_depth 3.0
"""


@pytest.fixture(scope="module")
def window() -> ModuleType:
    """Import the script as a module (benchmarks/ is not a package)."""
    spec = importlib.util.spec_from_file_location("server_window", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_metrics_skips_comments_and_keeps_labels(window: ModuleType) -> None:
    metrics = window.parse_metrics(BEFORE + "not a metric line\nbroken{ 1\nvllm_srun_ready NaNx\n")
    assert metrics[("vllm_srun_queue_depth", "")] == 0.0
    assert metrics[("vllm_srun_requests_total", 'endpoint="/v1/systemone",status="500"')] == 2.0
    assert ("vllm_srun_ready", "") not in metrics


def test_summarise_computes_rates_and_shares(window: ModuleType) -> None:
    summary = window.summarise(window.parse_metrics(BEFORE), window.parse_metrics(AFTER), 10.0)

    assert summary.requests_ok_per_second == pytest.approx(3.0)
    assert summary.questions_per_second == pytest.approx(6.0)
    assert summary.forwards_per_second == pytest.approx(4.0)
    assert summary.rows_per_forward == pytest.approx(1.5)
    assert summary.tokens_per_forward == pytest.approx(750.0)
    assert summary.tokens_per_second_wall == pytest.approx(3000.0)
    assert summary.tokens_per_second_in_forward == pytest.approx(3000.0)
    assert summary.forward_busy_fraction == pytest.approx(1.0)
    assert summary.mean_queue_wait_ms == pytest.approx(1000 * 8.2 / 34)
    assert summary.queue_depth == 3.0


def test_summarise_reports_failed_requests_by_status(window: ModuleType) -> None:
    summary = window.summarise(window.parse_metrics(BEFORE), window.parse_metrics(AFTER), 10.0)
    assert summary.failed_requests_by_status == {"500": 3, "429": 1}


def test_summarise_of_an_idle_window_is_all_zero(window: ModuleType) -> None:
    metrics = window.parse_metrics(BEFORE)
    summary = window.summarise(metrics, metrics, 20.0)

    assert summary.requests_ok_per_second == 0.0
    assert summary.rows_per_forward == 0.0
    assert summary.tokens_per_second_in_forward == 0.0
    assert summary.mean_queue_wait_ms == 0.0
    assert summary.failed_requests_by_status == {}


def test_format_mentions_failures(window: ModuleType) -> None:
    summary = window.summarise(window.parse_metrics(BEFORE), window.parse_metrics(AFTER), 10.0)
    assert "500: 3" in summary.format()
