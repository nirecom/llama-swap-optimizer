# llama-swap-optimizer

Automatically optimize [llama-swap](https://github.com/mostlygeek/llama-swap) configurations using [llama-optimus](https://github.com/BrunoArsioli/llama-optimus).

Runs llama-optimus benchmarks for each model defined in your llama-swap `config.yaml`, then generates an optimized config with tuned parameters (`-t`, `-b`, `-ub`, `-ngl`, `--flash-attn`, `--override-tensor`).

## How it works

1. Parses your llama-swap `config.yaml` and extracts GGUF model paths
2. Runs llama-optimus per model (3-stage optimization with Optuna)
3. Preserves your existing flags (`--ctx-size`, `--no-mmap`, `--jinja`, etc.)
4. Overwrites only the flags that llama-optimus optimizes
5. Generates `config-optimized.yaml` with the results (or overwrites `config.yaml` with `--overwrite`)

## Requirements

- Python 3.10+
- [llama.cpp](https://github.com/ggml-org/llama.cpp) binaries (`llama-bench` must be in the binary directory)

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/nirecom/llama-swap-optimizer.git
   cd llama-swap-optimizer
   ```

2. Install dependencies:

   ```bash
   # Using uv (recommended)
   uv sync

   # Or using pip
   pip install -r requirements.txt
   ```

3. Configure your environment:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your paths:

   ```
   LLAMA_BIN=/path/to/llama-server
   CONFIG_YAML=/path/to/llama-swap/config.yaml
   ```

   Results are saved to `./results/` by default (git-ignored).

## Usage

If you installed with **uv**, prefix commands with `uv run`:

```bash
uv run python llama_swap_optimizer.py [options]
```

If you installed with **pip** (with venv activated):

```bash
python llama_swap_optimizer.py [options]
```

### Examples

```bash
# Optimize all models
python llama_swap_optimizer.py

# Optimize a specific model
python llama_swap_optimizer.py --only my-large-model

# Skip specific models
python llama_swap_optimizer.py --skip my-large-model

# Preview without running (dry-run)
python llama_swap_optimizer.py --dry-run

# Apply saved results only (regenerate config without re-running benchmarks)
python llama_swap_optimizer.py --apply-only

# Overwrite config.yaml directly (backup created automatically)
python llama_swap_optimizer.py --apply-only --overwrite

# Custom trials / timeout
python llama_swap_optimizer.py --only my-large-model --trials 15 --timeout 43200
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `CONFIG_YAML` | Path to llama-swap config.yaml |
| `--llama-bin` | `LLAMA_BIN` | Path to llama.cpp binary directory |
| `--results-dir` | `./results/` | Directory to save per-model JSON results |
| `--trials` | 30 | Number of Optuna optimization trials |
| `--repeat` | 2 | Benchmark repetitions per trial |
| `--metric` | `tg` | Optimization target: `tg` (token generation), `pp` (prompt processing), `mean` |
| `--timeout` | 21600 (6h) | Timeout per model in seconds |
| `--only` | | Only optimize specified model(s) |
| `--skip` | | Skip specified model(s) |
| `--apply-only` | | Apply saved results without running benchmarks |
| `--dry-run` | | Preview what would be executed |
| `--overwrite` | | Overwrite the original config.yaml in place (backup created automatically) |
| `--output` | `config-optimized.yaml` | Output config file path |

## Skipping models with annotations

If a `model-annotations.yaml` file exists alongside your `config.yaml`, models with `skip_optimizer: true` are automatically excluded from optimization. This is useful for CPU-only models where GPU parameter tuning has no effect.

```yaml
# model-annotations.yaml
my-cpu-model:
  skip_optimizer: true
```

If the file does not exist, all models in `config.yaml` are optimized (backward compatible).

## Resume capability

Results are saved as JSON files in `./results/` (one per model). If optimization is interrupted, re-running the script will skip already-completed models. Delete a model's JSON file to force re-optimization.

## Output example

```
📊 Optimization Summary
============================================================
Model                           tok/s  threads  batch  ubatch  ngl  FA
------------------------------------------------------------
my-large-model-Q4_K_M           26.8       20   5728    3923   58   ✓
my-small-model-Q4_K_M          218.3       18   4976    6051   33   ✓
my-chat-model-Q8_0             146.4        8   2674    4423   46   ✓
my-code-model-Q4_K_M            58.3       20   8112    2254   71   ✓
```

## Related projects

- [llama-swap](https://github.com/mostlygeek/llama-swap) — Hot-swap multiple llama.cpp models behind a single API endpoint
- [llama-optimus](https://github.com/BrunoArsioli/llama-optimus) — Benchmark and auto-tune llama.cpp parameters with Optuna
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — LLM inference in C/C++

## License

MIT
