"""Tests for optimizer annotation support (optimizer: skip / lock)."""

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
        """No file -> empty dict, no error."""
        result = load_annotations(annotations_dir / "nonexistent")
        assert result == {}

    def test_loads_annotations_from_yaml(self, annotations_dir):
        """Normal case: reads annotations from yaml."""
        write_annotations(annotations_dir, {
            "model-a": {"role": "general", "optimizer": "skip"},
            "model-b": {"role": "judge"},
        })
        result = load_annotations(annotations_dir / "model-annotations.yaml")
        assert result["model-a"]["optimizer"] == "skip"
        assert "optimizer" not in result["model-b"]

    def test_returns_empty_dict_when_file_is_empty(self, annotations_dir):
        """Empty file -> empty dict."""
        (annotations_dir / "model-annotations.yaml").write_text("")
        result = load_annotations(annotations_dir / "model-annotations.yaml")
        assert result == {}


class TestFilterByAnnotations:
    """Tests for filter_by_annotations()."""

    def test_skip_excludes_from_optimize_and_apply(self):
        """optimizer: skip -> excluded from both optimize and apply."""
        models = ["model-a", "model-b"]
        annotations = {
            "model-a": {"optimizer": "skip"},
            "model-b": {"role": "general"},
        }
        assert filter_by_annotations(models, annotations, phase="optimize") == ["model-b"]
        assert filter_by_annotations(models, annotations, phase="apply") == ["model-b"]

    def test_lock_excludes_from_optimize_and_apply(self):
        """optimizer: lock -> excluded from both optimize and apply."""
        models = ["model-a", "model-b"]
        annotations = {
            "model-a": {"optimizer": "lock"},
            "model-b": {"role": "general"},
        }
        assert filter_by_annotations(models, annotations, phase="optimize") == ["model-b"]
        assert filter_by_annotations(models, annotations, phase="apply") == ["model-b"]

    def test_no_annotations_keeps_all_models(self):
        """Empty annotations -> all models kept."""
        models = ["model-a", "model-b"]
        result = filter_by_annotations(models, {}, phase="optimize")
        assert result == ["model-a", "model-b"]

    def test_model_not_in_annotations_is_kept(self):
        """Models absent from annotations are kept."""
        models = ["model-a", "model-new"]
        annotations = {
            "model-a": {"optimizer": "skip"},
        }
        result = filter_by_annotations(models, annotations, phase="optimize")
        assert result == ["model-new"]

    def test_no_optimizer_field_keeps_model(self):
        """Models with annotations but no optimizer field are kept."""
        models = ["model-a"]
        annotations = {
            "model-a": {"role": "general", "notes": "some notes"},
        }
        result = filter_by_annotations(models, annotations, phase="optimize")
        assert result == ["model-a"]

    def test_unknown_optimizer_value_keeps_model(self):
        """Unknown optimizer values don't trigger skip."""
        models = ["model-a"]
        annotations = {
            "model-a": {"optimizer": "unknown"},
        }
        result = filter_by_annotations(models, annotations, phase="optimize")
        assert result == ["model-a"]

    def test_backward_compat_skip_optimizer_true(self):
        """Legacy skip_optimizer: true still works."""
        models = ["model-a", "model-b"]
        annotations = {
            "model-a": {"skip_optimizer": True},
            "model-b": {"role": "general"},
        }
        assert filter_by_annotations(models, annotations, phase="optimize") == ["model-b"]
        assert filter_by_annotations(models, annotations, phase="apply") == ["model-b"]

    def test_default_phase_is_optimize(self):
        """Phase defaults to optimize."""
        models = ["model-a", "model-b"]
        annotations = {
            "model-a": {"optimizer": "skip"},
        }
        result = filter_by_annotations(models, annotations)
        assert result == ["model-b"]
