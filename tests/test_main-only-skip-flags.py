"""Tests for --only/--skip argparse flag accumulation (repeated flag usage)."""

import argparse
import sys

import pytest
import yaml

from llama_swap_optimizer import main


@pytest.fixture
def config_with_models(tmp_path):
    """Create a minimal config.yaml with three models and a results dir."""
    config = {
        "models": {
            "model-a": {"cmd": "llama-server --model /models/model-a.gguf"},
            "model-b": {"cmd": "llama-server --model /models/model-b.gguf"},
            "model-c": {"cmd": "llama-server --model /models/model-c.gguf"},
        }
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.dump(config, fh)

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    return config_path, results_dir


class TestOnlySkipFlagAccumulation:
    """Tests for repeated --only / --skip flag usage."""

    def test_only_repeated_flag_includes_both_models(self, config_with_models, monkeypatch, capsys):
        """Repeated --only flags should union targets, not overwrite to the last value."""
        config_path, results_dir = config_with_models
        monkeypatch.setattr(sys, "argv", [
            "prog",
            "--config", str(config_path),
            "--results-dir", str(results_dir),
            "--only", "model-a",
            "--only", "model-b",
            "--dry-run",
        ])

        main()

        out = capsys.readouterr().out
        assert "🔍 model-a" in out
        assert "🔍 model-b" in out
        assert "🔍 model-c" not in out

    def test_only_space_separated_still_works(self, config_with_models, monkeypatch, capsys):
        """Single --only flag with multiple space-separated values still works."""
        config_path, results_dir = config_with_models
        monkeypatch.setattr(sys, "argv", [
            "prog",
            "--config", str(config_path),
            "--results-dir", str(results_dir),
            "--only", "model-a", "model-b",
            "--dry-run",
        ])

        main()

        out = capsys.readouterr().out
        assert "🔍 model-a" in out
        assert "🔍 model-b" in out
        assert "🔍 model-c" not in out

    def test_skip_repeated_flag_excludes_both_models(self, config_with_models, monkeypatch, capsys):
        """Repeated --skip flags should union exclusions, not overwrite to the last value."""
        config_path, results_dir = config_with_models
        monkeypatch.setattr(sys, "argv", [
            "prog",
            "--config", str(config_path),
            "--results-dir", str(results_dir),
            "--skip", "model-a",
            "--skip", "model-b",
            "--dry-run",
        ])

        main()

        out = capsys.readouterr().out
        assert "🔍 model-a" not in out
        assert "🔍 model-b" not in out
        assert "🔍 model-c" in out

    def test_only_and_skip_combined_repeated_flags(self, config_with_models, monkeypatch, capsys):
        """Combining repeated --only and --skip flags applies both filters correctly."""
        config_path, results_dir = config_with_models
        monkeypatch.setattr(sys, "argv", [
            "prog",
            "--config", str(config_path),
            "--results-dir", str(results_dir),
            "--only", "model-a",
            "--only", "model-b",
            "--skip", "model-b",
            "--dry-run",
        ])

        main()

        out = capsys.readouterr().out
        assert "🔍 model-a" in out
        assert "🔍 model-b" not in out
        assert "🔍 model-c" not in out

    def test_only_unknown_model_yields_empty_targets(self, config_with_models, monkeypatch, capsys):
        """--only with a model name not in config silently produces no targets."""
        config_path, results_dir = config_with_models
        monkeypatch.setattr(sys, "argv", [
            "prog",
            "--config", str(config_path),
            "--results-dir", str(results_dir),
            "--only", "nonexistent-model",
            "--dry-run",
        ])

        main()

        out = capsys.readouterr().out
        assert "🔍 model-a" not in out
        assert "🔍 model-b" not in out
        assert "🔍 model-c" not in out

    def test_skip_all_models_yields_empty_targets(self, config_with_models, monkeypatch, capsys):
        """--skip covering all models produces no targets and no error."""
        config_path, results_dir = config_with_models
        monkeypatch.setattr(sys, "argv", [
            "prog",
            "--config", str(config_path),
            "--results-dir", str(results_dir),
            "--skip", "model-a",
            "--skip", "model-b",
            "--skip", "model-c",
            "--dry-run",
        ])

        main()

        out = capsys.readouterr().out
        assert "🔍 model-a" not in out
        assert "🔍 model-b" not in out
        assert "🔍 model-c" not in out

    def test_annotation_excluded_model_not_resurrected_by_only(self, tmp_path, monkeypatch, capsys):
        """--only cannot resurrect a model excluded by optimizer: skip annotation."""
        config = {
            "models": {
                "model-skip": {"cmd": "llama-server --model /models/model-skip.gguf"},
                "model-ok": {"cmd": "llama-server --model /models/model-ok.gguf"},
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as fh:
            yaml.dump(config, fh)

        annotations = {"model-skip": {"optimizer": "skip"}}
        with open(tmp_path / "model-annotations.yaml", "w", encoding="utf-8") as fh:
            yaml.dump(annotations, fh)

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        monkeypatch.setattr(sys, "argv", [
            "prog",
            "--config", str(config_path),
            "--results-dir", str(results_dir),
            "--only", "model-skip",
            "--dry-run",
        ])

        main()

        out = capsys.readouterr().out
        assert "🔍 model-skip" not in out


class TestArgparseAccumulation:
    """Direct argparse-level tests pinning the action='extend' fix."""

    def test_only_flag_accumulates_not_overwrites(self):
        """action='extend' on --only causes repeated flags to accumulate."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--only", nargs="+", action="extend", default=[])
        args = parser.parse_args(["--only", "model-a", "--only", "model-b"])
        assert args.only == ["model-a", "model-b"]

    def test_skip_flag_accumulates_not_overwrites(self):
        """action='extend' on --skip causes repeated flags to accumulate."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--skip", nargs="+", action="extend", default=[])
        args = parser.parse_args(["--skip", "model-a", "--skip", "model-b"])
        assert args.skip == ["model-a", "model-b"]
