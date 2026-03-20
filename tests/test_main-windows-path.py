"""Tests for Windows path compatibility fix in run_llama_optimus."""

import subprocess
from unittest.mock import patch, MagicMock

import pytest

from llama_swap_optimizer import run_llama_optimus


@pytest.fixture
def mock_subprocess():
    """Mock subprocess.run to capture the command without executing."""
    with patch("llama_swap_optimizer.subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        yield mock_run


def _get_called_args(mock_run) -> list[str]:
    """Extract the command list from mock call."""
    return mock_run.call_args[0][0]


def _extract_paths(cmd: list[str]) -> tuple[str, str]:
    """Extract --llama-bin and --model values from command list."""
    llama_bin = cmd[cmd.index("--llama-bin") + 1]
    model = cmd[cmd.index("--model") + 1]
    return llama_bin, model


# --- Normal cases ---

class TestWindowsPathConversion:
    def test_backslashes_converted_to_forward(self, mock_subprocess):
        run_llama_optimus(
            model_path=r"C:\models\test.gguf",
            trials=1, repeat=1, metric="tg",
            llama_bin=r"C:\llama\llama-bench.exe",
        )
        llama_bin, model = _extract_paths(_get_called_args(mock_subprocess))
        assert "\\" not in llama_bin
        assert "\\" not in model
        assert llama_bin == "C:/llama/llama-bench.exe"
        assert model == "C:/models/test.gguf"

    def test_unix_paths_unchanged(self, mock_subprocess):
        run_llama_optimus(
            model_path="/models/test.gguf",
            trials=1, repeat=1, metric="tg",
            llama_bin="/usr/bin/llama-bench",
        )
        llama_bin, model = _extract_paths(_get_called_args(mock_subprocess))
        assert llama_bin == "/usr/bin/llama-bench"
        assert model == "/models/test.gguf"


# --- Edge cases ---

class TestPathEdgeCases:
    def test_mixed_separators(self, mock_subprocess):
        run_llama_optimus(
            model_path=r"C:\models/sub\test.gguf",
            trials=1, repeat=1, metric="tg",
            llama_bin=r"C:\llama/bin\llama-bench.exe",
        )
        llama_bin, model = _extract_paths(_get_called_args(mock_subprocess))
        assert "\\" not in llama_bin
        assert "\\" not in model
        assert llama_bin == "C:/llama/bin/llama-bench.exe"
        assert model == "C:/models/sub/test.gguf"

    def test_no_backslashes_unchanged(self, mock_subprocess):
        run_llama_optimus(
            model_path="models/test.gguf",
            trials=1, repeat=1, metric="tg",
            llama_bin="llama-bench",
        )
        llama_bin, model = _extract_paths(_get_called_args(mock_subprocess))
        assert llama_bin == "llama-bench"
        assert model == "models/test.gguf"

    def test_empty_paths(self, mock_subprocess):
        run_llama_optimus(
            model_path="",
            trials=1, repeat=1, metric="tg",
            llama_bin="",
        )
        llama_bin, model = _extract_paths(_get_called_args(mock_subprocess))
        assert llama_bin == ""
        assert model == ""

    def test_unc_path(self, mock_subprocess):
        run_llama_optimus(
            model_path=r"\\server\share\models\test.gguf",
            trials=1, repeat=1, metric="tg",
            llama_bin=r"\\server\share\bin\llama-bench.exe",
        )
        llama_bin, model = _extract_paths(_get_called_args(mock_subprocess))
        assert "\\" not in llama_bin
        assert "\\" not in model
        assert llama_bin == "//server/share/bin/llama-bench.exe"
        assert model == "//server/share/models/test.gguf"

    def test_special_chars_in_path(self, mock_subprocess):
        run_llama_optimus(
            model_path=r"C:\Program Files (x86)\models\[draft]\test.gguf",
            trials=1, repeat=1, metric="tg",
            llama_bin=r"C:\Program Files (x86)\llama\llama-bench.exe",
        )
        llama_bin, model = _extract_paths(_get_called_args(mock_subprocess))
        assert "\\" not in llama_bin
        assert "\\" not in model
        assert llama_bin == "C:/Program Files (x86)/llama/llama-bench.exe"
        assert model == "C:/Program Files (x86)/models/[draft]/test.gguf"

    def test_relative_path_with_parent_refs(self, mock_subprocess):
        run_llama_optimus(
            model_path=r"..\..\models\test.gguf",
            trials=1, repeat=1, metric="tg",
            llama_bin=r"..\bin\llama-bench.exe",
        )
        llama_bin, model = _extract_paths(_get_called_args(mock_subprocess))
        assert "\\" not in llama_bin
        assert "\\" not in model
        assert llama_bin == "../bin/llama-bench.exe"
        assert model == "../../models/test.gguf"

    def test_trailing_separator(self, mock_subprocess):
        run_llama_optimus(
            model_path=r"C:\models\test.gguf",
            trials=1, repeat=1, metric="tg",
            llama_bin="C:\\llama\\bin\\",
        )
        llama_bin, model = _extract_paths(_get_called_args(mock_subprocess))
        assert "\\" not in llama_bin
        assert llama_bin == "C:/llama/bin/"


# --- Idempotency cases ---

class TestIdempotency:
    def test_conversion_is_idempotent(self, mock_subprocess):
        """Converting an already-converted path should produce the same result."""
        run_llama_optimus(
            model_path="C:/models/test.gguf",
            trials=1, repeat=1, metric="tg",
            llama_bin="C:/llama/llama-bench.exe",
        )
        llama_bin, model = _extract_paths(_get_called_args(mock_subprocess))
        assert llama_bin == "C:/llama/llama-bench.exe"
        assert model == "C:/models/test.gguf"

    def test_repeated_calls_produce_same_command(self, mock_subprocess):
        """Two calls with the same args should produce identical commands."""
        args = dict(
            model_path=r"C:\models\test.gguf",
            trials=1, repeat=1, metric="tg",
            llama_bin=r"C:\llama\llama-bench.exe",
        )
        run_llama_optimus(**args)
        cmd1 = _get_called_args(mock_subprocess)
        mock_subprocess.reset_mock()

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        run_llama_optimus(**args)
        cmd2 = _get_called_args(mock_subprocess)
        assert cmd1 == cmd2
