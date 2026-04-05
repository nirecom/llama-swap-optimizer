"""Tests for skip_optimizer annotation support."""

import tempfile
from pathlib import Path

import pytest
import yaml

from llama_swap_optimizer import load_annotations, filter_by_annotations


@pytest.fixture
def annotations_dir():
    """Create a temp directory with a model-annotations.yaml file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def write_annotations(path, data):
    """Helper to write a model-annotations.yaml file."""
    with open(path / "model-annotations.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data, f)


class TestLoadAnnotations:
    """Tests for load_annotations()."""

    def test_returns_empty_dict_when_file_missing(self, annotations_dir):
        """No file → empty dict, no error."""
        result = load_annotations(annotations_dir / "nonexistent")
        assert result == {}

    def test_loads_annotations_from_yaml(self, annotations_dir):
        """Normal case: reads annotations from yaml."""
        write_annotations(annotations_dir, {
            "model-a": {"role": "general", "skip_optimizer": True},
            "model-b": {"role": "judge"},
        })
        result = load_annotations(annotations_dir / "model-annotations.yaml")
        assert result["model-a"]["skip_optimizer"] is True
        assert "skip_optimizer" not in result["model-b"]

    def test_returns_empty_dict_when_file_is_empty(self, annotations_dir):
        """Empty file → empty dict."""
        (annotations_dir / "model-annotations.yaml").write_text("")
        result = load_annotations(annotations_dir / "model-annotations.yaml")
        assert result == {}


class TestFilterByAnnotations:
    """Tests for filter_by_annotations()."""

    def test_skips_models_with_skip_optimizer_true(self):
        """Models with skip_optimizer: true are excluded."""
        models = ["model-a", "model-b", "model-c"]
        annotations = {
            "model-a": {"skip_optimizer": True},
            "model-b": {"role": "general"},
            "model-c": {"skip_optimizer": False},
        }
        result = filter_by_annotations(models, annotations)
        assert result == ["model-b", "model-c"]

    def test_no_annotations_keeps_all_models(self):
        """Empty annotations → all models kept."""
        models = ["model-a", "model-b"]
        result = filter_by_annotations(models, {})
        assert result == ["model-a", "model-b"]

    def test_model_not_in_annotations_is_kept(self):
        """Models absent from annotations are kept."""
        models = ["model-a", "model-new"]
        annotations = {
            "model-a": {"skip_optimizer": True},
        }
        result = filter_by_annotations(models, annotations)
        assert result == ["model-new"]

    def test_skip_optimizer_only_when_true(self):
        """Only skip_optimizer: true triggers skip, not other truthy values."""
        models = ["model-a", "model-b"]
        annotations = {
            "model-a": {"skip_optimizer": "yes"},  # truthy but not True
            "model-b": {"skip_optimizer": True},
        }
        # "yes" is truthy, so it should also skip
        result = filter_by_annotations(models, annotations)
        assert "model-b" not in result
