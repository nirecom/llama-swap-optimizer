### FEATURE: Initial Implementation (2026-02-22, d64fafe, #1)
Background: Initial implementation of llama-swap config optimizer using llama-optimus
Changes: Core optimizer working end-to-end with llama-optimus integration

### BUGFIX: Windows Path Compatibility Fix (2026-03-21, 819653f, #2)
Background: llama-optimus uses shlex.split() internally, treating backslashes as escape characters. Windows native paths like C:/models/file.gguf caused parsing failures.
Changes: Convert backslashes to forward slashes in llama_bin and model_path before invoking llama-optimus
Test gap: No test for Windows path handling in llama-optimus invocation

### FEATURE: PR #5 — fix/fix-only-flag-accumulation (2026-07-29, ca787f2c730c037f5e3e23528a75c4e67bfa3def, #5)
Background: fix(cli): accumulate --only/--skip flags across repeated invocations
Changes: Fix bug where `--only ModelA --only ModelB` only targeted ModelB: argparse `nargs="+"` without `action="extend"` caused repeated flags to overwrite instead of accumulate; added `action="extend"` to `--only` and `--skip` definitions, and added 9 unit tests covering accumulation, backward compatibility, edge cases, and annotation interaction (#4) <!-- compose-doc-append-sentinel: branch=fix/fix-only-flag-accumulation pr=#5 -->
