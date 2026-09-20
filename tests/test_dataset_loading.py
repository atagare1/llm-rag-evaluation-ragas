import json

import pytest

from utils import read_test_data


def test_read_test_data_uses_supplied_filename(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"question": "q1", "reference": "r1"}), encoding="utf-8")
    second.write_text(json.dumps({"question": "q2", "reference": "r2"}), encoding="utf-8")

    assert read_test_data("first.json", base_dir=tmp_path)["question"] == "q1"
    assert read_test_data("second.json", base_dir=tmp_path)["question"] == "q2"


def test_read_test_data_loads_checked_in_samples_by_filename():
    primary = read_test_data("rag_test_data.json")
    faithfulness = read_test_data("rag_test_data_faithfulness.json")

    assert primary["question"]
    assert "reference" in primary
    assert faithfulness["question"]
    assert "reference" in faithfulness


def test_read_test_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_test_data("does-not-exist.json", base_dir=tmp_path)
