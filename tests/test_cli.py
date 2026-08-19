"""Unit tests for the CLI recommender interface."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from cli_recommender import main


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out, _ = capsys.readouterr()
    assert "Content-based Steam game recommender" in out


def test_cli_search(engine, capsys):
    ret = main(["--search", "alpha", "-k", "3", "--no-faiss"], engine=engine)
    assert ret == 0
    out, _ = capsys.readouterr()
    assert "Search results for 'alpha'" in out


def test_cli_genres(engine, capsys):
    ret = main(["--genres", "--no-faiss"], engine=engine)
    assert ret == 0
    out, _ = capsys.readouterr()
    assert "Available genres:" in out


def test_cli_json_recommendation(engine, capsys):
    ret = main(["Alpha Shooter", "-k", "2", "--min-reviews", "100", "--min-rating", "0.5", "--json", "--no-faiss"], engine=engine)
    assert ret == 0
    out, _ = capsys.readouterr()
    data = json.loads(out)
    assert isinstance(data, list)
    assert len(data) <= 2


def test_cli_export_json(engine, tmp_path: Path, capsys):
    out_file = tmp_path / "test_recs.json"
    ret = main([
        "Alpha Shooter",
        "-k", "2",
        "--min-reviews", "100",
        "--min-rating", "0.5",
        "--export-json", str(out_file),
        "--no-faiss",
    ], engine=engine)
    assert ret == 0
    assert out_file.exists()
    with open(out_file, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)


def test_cli_no_args(engine, capsys):
    ret = main([], engine=engine)
    assert ret == 1
