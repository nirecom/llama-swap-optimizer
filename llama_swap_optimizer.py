#!/usr/bin/env python3
"""
llama-swap-optimizer
====================
Automatically optimize llama-swap config.yaml using llama-optimus.
Runs llama-optimus for each model to find optimal llama.cpp inference parameters,
then generates an optimized config file.

Usage:
  python llama_swap_optimizer.py                          # optimize all models
  python llama_swap_optimizer.py --only my-large-model      # optimize specific model(s)
  python llama_swap_optimizer.py --skip my-large-model      # skip specific model(s)
  python llama_swap_optimizer.py --apply-only              # apply saved results only (no run)
  python llama_swap_optimizer.py --dry-run                 # preview what would be executed

Configuration:
  Copy .env.example to .env and edit paths to match your environment.
"""

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from llama_optimus.override_patterns import OVERRIDE_PATTERNS
import yaml

# Load .env from the same directory as this script
load_dotenv(Path(__file__).resolve().parent / ".env")

# ============================================================
# CONFIG - Override via .env file or environment variables
# ============================================================
LLAMA_BIN = os.environ.get("LLAMA_BIN", "")
CONFIG_YAML = os.environ.get("CONFIG_YAML", "")
RESULTS_DIR = os.environ.get("RESULTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))

DEFAULT_TRIALS = int(os.environ.get("DEFAULT_TRIALS", "30"))
DEFAULT_REPEAT = int(os.environ.get("DEFAULT_REPEAT", "2"))
DEFAULT_METRIC = os.environ.get("DEFAULT_METRIC", "tg")
DEFAULT_TIMEOUT = int(os.environ.get("DEFAULT_TIMEOUT", "21600"))  # 6 hours
DEFAULT_OVERWRITE = os.environ.get("DEFAULT_OVERWRITE", "").lower() in ("1", "true", "yes")
# ============================================================


# Flags that llama-optimus optimizes (these will be overwritten with results)
OPTIMIZED_FLAGS = {
    "-t", "--threads",
    "-b", "--batch-size",
    "-ub", "--ubatch-size",
    "-ngl", "--n-gpu-layers",
    "--flash-attn",
    "--override-tensor",
}

# Boolean flags (no value required)
BOOLEAN_FLAGS = {"--no-mmap", "--jinja", "--verbose", "--no-webui"}


def parse_cmd_flags(cmd: str) -> tuple[str, str, str, dict]:
    """
    Parse a llama-swap cmd string and return:
    (exe_path, model_path, extra_flags_dict)

    extra_flags_dict: flag_name -> value (boolean flags have value True)
    """
    # YAML multiline strings are collapsed into spaces
    cmd = " ".join(cmd.split())

    # Tokenize (handles Windows backslash paths)
    tokens = cmd.split()

    exe_path = tokens[0]
    flags = {}
    model_path = ""
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--model":
            # Model path may be quoted
            i += 1
            model_path = tokens[i].strip('"').strip("'")
            # Rejoin tokens split inside quoted paths
            while model_path.count('"') % 2 != 0 and i + 1 < len(tokens):
                i += 1
                model_path += " " + tokens[i].strip('"')
        elif tok.startswith("-"):
            if tok in BOOLEAN_FLAGS:
                # Handle --flag on/off style values
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    val = tokens[i + 1]
                    if val.lower() in ("on", "1", "true"):
                        flags[tok] = True
                        i += 1
                    elif val.lower() in ("off", "0", "false"):
                        flags[tok] = False
                        i += 1
                    else:
                        flags[tok] = True
                else:
                    flags[tok] = True
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                flags[tok] = tokens[i + 1].strip('"').strip("'")
                i += 1
            else:
                flags[tok] = True
        i += 1

    return exe_path, model_path, flags


def build_cmd(exe_path: str, model_path: str, flags: dict, line_width: int = 100) -> str:
    """
    Rebuild a llama-swap cmd string from parsed parts.
    Wraps flags across multiple lines for readability.
    """
    indent = "  "

    # Build flag pairs (--host, --port, --model handled separately)
    flag_pairs = []
    skip = {"--host", "--port", "--model"}
    for flag, val in sorted(flags.items()):
        if flag in skip:
            continue
        if val is True:
            flag_pairs.append(flag)
        elif val is not False:
            if " " in str(val) or "\\" in str(val):
                flag_pairs.append(f'{flag} "{val}"')
            else:
                flag_pairs.append(f"{flag} {val}")

    # Quote model path if it contains spaces
    model_str = f'"{model_path}"' if " " in model_path else model_path

    lines = [
        f"{exe_path}",
        f"{indent}--host 127.0.0.1 --port ${{PORT}}",
        f"{indent}--model {model_str}",
    ]

    # Pack remaining flags into lines within line_width
    current_line = indent
    for pair in flag_pairs:
        if current_line == indent:
            current_line += pair
        elif len(current_line) + 1 + len(pair) > line_width:
            lines.append(current_line)
            current_line = indent + pair
        else:
            current_line += " " + pair
    if current_line != indent:
        lines.append(current_line)

    return "\n".join(lines)


def extract_model_path_from_cmd(cmd: str) -> str:
    """Extract the GGUF model path from a cmd string."""
    _, model_path, _ = parse_cmd_flags(cmd)
    return model_path


def run_llama_optimus(model_path: str, trials: int, repeat: int, metric: str,
                      llama_bin: str = LLAMA_BIN, timeout: int = DEFAULT_TIMEOUT) -> dict | None:
    """
    Run llama-optimus and return optimization results.
    Returns: {"stage2": {...}, "stage3": {...}, "tokens_per_sec": float, "raw_output": str}
    """
    # Convert backslashes to forward slashes for llama-optimus compatibility
    # (its internal shlex.split() treats backslashes as escape characters)
    llama_bin = llama_bin.replace("\\", "/")
    model_path = model_path.replace("\\", "/")

    cmd = [
        sys.executable, "-m", "llama_optimus.cli",
        "--llama-bin", llama_bin,
        "--model", model_path,
        "--metric", metric,
        "--trials", str(trials),
        "--repeat", str(repeat),
    ]

    print(f"  Running: {' '.join(cmd)}")
    print(f"  (This may take several minutes to hours...)")
    print()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"  ❌ Timed out ({timeout // 3600}h {(timeout % 3600) // 60}m)")
        print(f"     Use --timeout to increase, or --trials to reduce workload")
        return None
    except FileNotFoundError:
        print("  ❌ llama-optimus not found. Verify: pip install llama-optimus")
        return None

    output = result.stdout + "\n" + result.stderr
    print(output[-2000:])  # Show last 2000 chars

    if result.returncode != 0:
        print(f"  ❌ llama-optimus exited with error (code={result.returncode})")
        return None

    # Parse Stage 2 and Stage 3 results
    stage2 = parse_best_config(output, "Stage_2")
    stage3 = parse_best_config(output, "Stage_3")
    tps = parse_best_tps(output, "Stage_3", metric)

    if not stage3:
        print("  ❌ Failed to parse Stage_3 results")
        return None

    return {
        "stage2": stage2 or {"flash_attn": 0, "override_tensor": "none"},
        "stage3": stage3,
        "tokens_per_sec": tps,
        "raw_output": output,
    }


def parse_best_config(output: str, stage: str) -> dict | None:
    """Parse 'Best config Stage_X: {...}' from llama-optimus output."""
    pattern = rf"Best config {stage}:\s*(\{{.*?\}})"
    match = re.search(pattern, output)
    if match:
        try:
            return ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError):
            return None
    return None


def parse_best_tps(output: str, stage: str, metric: str) -> float | None:
    """Parse 'Best Stage_X tg tokens/sec: 73.5' from llama-optimus output."""
    pattern = rf"Best {stage} {metric} tokens/sec:\s*([\d.]+)"
    match = re.search(pattern, output)
    if match:
        return float(match.group(1))
    return None


def apply_optimization(flags: dict, result: dict) -> dict:
    """
    Apply optimization results to existing flags dict.
    Only overwrites flags that llama-optimus optimizes; preserves all others.
    """
    new_flags = {}
    for k, v in flags.items():
        if k not in OPTIMIZED_FLAGS:
            new_flags[k] = v

    s3 = result["stage3"]
    s2 = result["stage2"]

    new_flags["-t"] = str(s3["threads"])
    new_flags["-b"] = str(s3["batch"])
    new_flags["-ub"] = str(s3["u_batch"])
    new_flags["-ngl"] = str(s3["gpu_layers"])

    # llama-server expects an explicit value: --flash-attn on
    if s2.get("flash_attn") == 1:
        new_flags["--flash-attn"] = "on"
    # If flash_attn == 0, omit the flag entirely

    # Resolve override_tensor key name to actual pattern string
    override = s2.get("override_tensor", "none")
    if override and override != "none":
        pattern = OVERRIDE_PATTERNS.get(override, override)
        if pattern:
            new_flags["--override-tensor"] = pattern

    return new_flags


def load_results(results_dir: str) -> dict:
    """Load previously saved optimization results."""
    results = {}
    rdir = Path(results_dir)
    if rdir.exists():
        for f in rdir.glob("*.json"):
            model_name = f.stem
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                results[model_name] = data
    return results


def save_result(results_dir: str, model_name: str, result: dict):
    """Save optimization result as JSON."""
    rdir = Path(results_dir)
    rdir.mkdir(parents=True, exist_ok=True)

    # Save raw_output separately (keeps JSON small)
    raw = result.pop("raw_output", "")
    out_path = rdir / f"{model_name}.json"

    save_data = {
        **result,
        "model_name": model_name,
        "optimized_at": datetime.now().isoformat(),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(save_data, fh, indent=2, ensure_ascii=False)

    # Save full log
    log_path = rdir / f"{model_name}.log"
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(raw)

    print(f"  ✅ Results saved: {out_path}")


def generate_optimized_config(config: dict, results: dict) -> dict:
    """Apply optimization results to config and return a new config dict."""
    new_config = deepcopy(config)

    for model_name, model_conf in new_config.get("models", {}).items():
        if model_name not in results:
            print(f"  ⏭️  {model_name}: no results available (skipped)")
            continue

        result = results[model_name]
        cmd = model_conf["cmd"]
        exe_path, model_path, flags = parse_cmd_flags(cmd)

        # Apply optimized flags
        new_flags = apply_optimization(flags, result)

        # Rebuild cmd string
        new_cmd = build_cmd(exe_path, model_path, new_flags)
        model_conf["cmd"] = new_cmd

        tps = result.get("tokens_per_sec", "?")
        print(f"  ✅ {model_name}: optimized ({tps} tok/s)")

    return new_config


def write_config(config: dict, output_path: str):
    """Write config dict as a YAML file."""

    # Custom dumper to preserve multiline cmd strings
    class MultilineDumper(yaml.Dumper):
        pass

    def str_representer(dumper, data):
        if "\n" in data:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    MultilineDumper.add_representer(str, str_representer)

    with open(output_path, "w", encoding="utf-8") as fh:
        yaml.dump(
            config,
            fh,
            Dumper=MultilineDumper,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )


def main():
    parser = argparse.ArgumentParser(
        description="llama-swap-optimizer: auto-optimize llama-swap config.yaml using llama-optimus"
    )
    parser.add_argument("--config", default=CONFIG_YAML,
                        help=f"Path to llama-swap config.yaml (default: {CONFIG_YAML})")
    parser.add_argument("--llama-bin", default=LLAMA_BIN,
                        help=f"Path to llama.cpp binary directory (default: {LLAMA_BIN})")
    parser.add_argument("--results-dir", default=RESULTS_DIR,
                        help=f"Directory to save results (default: {RESULTS_DIR})")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help=f"Number of llama-optimus trials (default: {DEFAULT_TRIALS})")
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT,
                        help=f"Repetitions per trial (default: {DEFAULT_REPEAT})")
    parser.add_argument("--metric", default=DEFAULT_METRIC, choices=["tg", "pp", "mean"],
                        help=f"Optimization target metric (default: {DEFAULT_METRIC})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"Timeout per model in seconds (default: {DEFAULT_TIMEOUT} = {DEFAULT_TIMEOUT//3600}h)")
    parser.add_argument("--only", nargs="+", default=[],
                        help="Only optimize specified model(s)")
    parser.add_argument("--skip", nargs="+", default=[],
                        help="Skip specified model(s)")
    parser.add_argument("--apply-only", action="store_true",
                        help="Apply saved results only (do not run llama-optimus)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be executed without running")
    parser.add_argument("--overwrite", action="store_true", default=DEFAULT_OVERWRITE,
                        help="Overwrite the original config.yaml in place (backup created automatically)")
    parser.add_argument("--output", default=None,
                        help="Output config path (default: config-optimized.yaml)")

    args = parser.parse_args()

    llama_bin = args.llama_bin

    # Load config.yaml
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    models = config.get("models", {})
    print(f"📋 Found {len(models)} model(s):")
    for name in models:
        print(f"   - {name}")
    print()

    # Filter target models
    target_models = list(models.keys())
    if args.only:
        target_models = [m for m in target_models if m in args.only]
    if args.skip:
        target_models = [m for m in target_models if m not in args.skip]

    # Load existing results
    results = load_results(args.results_dir)
    existing = [m for m in target_models if m in results]
    if existing:
        print(f"📂 Existing results found: {', '.join(existing)}")
        print()

    # Phase 1: Run llama-optimus
    if not args.apply_only:
        for model_name in target_models:
            cmd = models[model_name]["cmd"]
            model_path = extract_model_path_from_cmd(cmd)

            if not model_path:
                print(f"⚠️  {model_name}: could not extract model path (skipped)")
                continue

            if model_name in results and not args.dry_run:
                prev_tps = results[model_name].get("tokens_per_sec", "?")
                print(f"⏭️  {model_name}: already optimized ({prev_tps} tok/s) -> delete result file to re-run")
                continue

            print(f"{'🔍' if args.dry_run else '🚀'} {model_name}")
            print(f"   Model: {model_path}")
            print(f"   trials={args.trials}, repeat={args.repeat}, metric={args.metric}, timeout={args.timeout}s")

            if args.dry_run:
                print(f"   (dry-run: skipping execution)")
                print()
                continue

            result = run_llama_optimus(model_path, args.trials, args.repeat, args.metric, llama_bin, args.timeout)
            if result:
                save_result(args.results_dir, model_name, result)
                results[model_name] = result
                print()
            else:
                print(f"   ❌ Optimization failed")
                print()

    if args.dry_run:
        print("(dry-run mode: exiting)")
        return

    # Phase 2: Generate optimized config.yaml
    print()
    print("=" * 60)
    print("📝 Applying optimization results to config.yaml...")
    print("=" * 60)

    # Reload results (raw_output was stripped during save)
    results = load_results(args.results_dir)

    new_config = generate_optimized_config(config, results)

    # Determine output path
    if args.output:
        output_path = args.output
    elif args.overwrite:
        output_path = str(config_path)
    else:
        output_path = str(config_path.parent / "config-optimized.yaml")

    # Backup before overwriting
    if Path(output_path).resolve() == config_path.resolve():
        backup = str(config_path) + f".bak.{datetime.now():%Y%m%d_%H%M%S}"
        shutil.copy2(config_path, backup)
        print(f"💾 Backup created: {backup}")

    write_config(new_config, output_path)
    print()
    print(f"✅ Optimized config written to: {output_path}")

    if Path(output_path).resolve() != config_path.resolve():
        print()
        print("Next steps:")
        print(f"  1. Review {output_path}")
        print(f"  2. If it looks good:")
        print(f"     copy \"{output_path}\" \"{config_path}\"")
        print(f"  3. Restart llama-swap")
    else:
        print("   (original config overwritten, restart llama-swap to apply)")

    # Summary
    print()
    print("=" * 60)
    print("📊 Optimization Summary")
    print("=" * 60)
    print(f"{'Model':<30} {'tok/s':>8}  {'threads':>7} {'batch':>6} {'ubatch':>7} {'ngl':>4} {'FA':>3}")
    print("-" * 60)
    for model_name in models:
        if model_name in results:
            r = results[model_name]
            s3 = r.get("stage3", {})
            s2 = r.get("stage2", {})
            tps = r.get("tokens_per_sec", 0)
            fa = "✓" if s2.get("flash_attn") == 1 else "-"
            print(
                f"{model_name:<30} {tps:>8.1f}  "
                f"{s3.get('threads', '?'):>7} {s3.get('batch', '?'):>6} "
                f"{s3.get('u_batch', '?'):>7} {s3.get('gpu_layers', '?'):>4} {fa:>3}"
            )
        else:
            print(f"{model_name:<30} {'(pending)':>8}")


if __name__ == "__main__":
    main()
