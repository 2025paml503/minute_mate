#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p data experiments notebooks

python -m scripts.explore_data
python -m scripts.model_walkthrough
python -m scripts.prepare_data
python -m scripts.run_e1_baseline
python -m scripts.run_e2_prompts
python -m scripts.run_e3_decoding
python -m scripts.run_e4_structured
python -m scripts.run_e5_tools
python -m scripts.evaluate
python -m scripts.build_notebook