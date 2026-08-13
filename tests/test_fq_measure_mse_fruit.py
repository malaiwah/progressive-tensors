"""Pure orchestration tests for fq_measure_mse_fruit."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import fq_measure_mse_fruit as measure


def test_stratified_sampling_includes_each_stage_chain():
    stages = [
        {"label": "res1", "experts": "all"},
        {"label": "res2", "experts": [0, 1]},
    ]
    tiers = measure.select_stratified_experts([0, 1, 2, 3], stages, 1)
    assert tiers == {"res1+res2": [0], "res1": [2]}


def test_parse_ids_rejects_duplicates_and_negative_values():
    assert measure.parse_ids("3,1") == [3, 1]
    with pytest.raises(Exception, match="unique non-negative"):
        measure.parse_ids("1,1")
    with pytest.raises(Exception, match="unique non-negative"):
        measure.parse_ids("-1")


def test_atomic_output_creates_parent_and_replaces_file(tmp_path: Path):
    output = tmp_path / "nested" / "results.json"
    measure.write_json_atomic(output, {"version": 1})
    measure.write_json_atomic(output, {"schema": "fq-msrt-mse/1"})
    assert json.loads(output.read_text()) == {"schema": "fq-msrt-mse/1"}
    assert not (output.parent / ".results.json.tmp").exists()
