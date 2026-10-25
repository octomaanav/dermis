#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${HOME}/.platformio/penv/bin/pio" device monitor --baud 115200 "${@}"
