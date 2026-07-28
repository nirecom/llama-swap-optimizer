### FEATURE: Initial Implementation (2026-02-22, d64fafe, #1)
Background: Initial implementation of llama-swap config optimizer using llama-optimus
Changes: Core optimizer working end-to-end with llama-optimus integration

### BUGFIX: Windows Path Compatibility Fix (2026-03-21, 819653f, #2)
Background: llama-optimus uses shlex.split() internally, treating backslashes as escape characters. Windows native paths like C:/models/file.gguf caused parsing failures.
Changes: Convert backslashes to forward slashes in llama_bin and model_path before invoking llama-optimus
Test gap: No test for Windows path handling in llama-optimus invocation
