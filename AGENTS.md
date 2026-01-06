# Repository Guidelines

## Project Structure & Module Organization
This repository currently contains a minimal setup with only `README.md` at the root. No source, tests, or assets are present yet. When adding code, keep it organized and predictable (for example, place application code under `src/`, tests under `tests/`, and small data/assets under `assets/`).

## Build, Test, and Development Commands
No build or runtime commands are defined in the repository at this time. If you add tooling, document the exact commands here with short explanations (for example, `python -m sermon_transcribe` to run locally, or `pytest` to execute tests).

## Coding Style & Naming Conventions
No formatting or linting tools are configured yet. Until tooling is added:
- Use 2 spaces for Markdown lists and 4 spaces for code blocks.
- Prefer `snake_case` for filenames and functions if using Python.
- Keep script names action-oriented (for example, `transcribe_audio.py`).
If you introduce a formatter or linter, specify the command (for example, `ruff format` or `black .`) and adhere to it consistently.

## Testing Guidelines
No test framework or coverage requirements are configured. If tests are added, place them under `tests/` and name files with a clear suffix (for example, `test_transcription.py`). Document the test runner command and any required test data.

## Commit & Pull Request Guidelines
The Git history only includes an initial commit, so no established convention exists. Use concise, imperative commit subjects (for example, "Add transcription CLI") and include a short body when behavior changes are non-obvious. For pull requests, include a summary of changes, any relevant usage notes, and sample inputs/outputs if the behavior affects transcription results.

## Configuration & Secrets
Avoid committing credentials or API keys. If configuration is needed, prefer environment variables and document them in `README.md` (for example, `WHISPER_MODEL=large-v3`).
