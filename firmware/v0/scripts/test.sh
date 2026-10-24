#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${project_dir}/scripts/build.sh"
PYTHONPATH="${project_dir}/tools" python3 -m unittest discover -s "${project_dir}/tests" -v

