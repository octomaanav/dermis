#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if command -v pio >/dev/null 2>&1; then
  pio_bin="$(command -v pio)"
elif [[ -x "${HOME}/.platformio/penv/bin/pio" ]]; then
  pio_bin="${HOME}/.platformio/penv/bin/pio"
else
  echo "PlatformIO Core was not found." >&2
  echo "Install it from https://platformio.org/install/cli and retry." >&2
  exit 127
fi

cd "${project_dir}"
exec "${pio_bin}" "$@"

