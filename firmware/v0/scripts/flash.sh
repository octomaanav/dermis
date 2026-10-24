#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <serial-port>" >&2
  echo "Example: $0 /dev/cu.usbserial-0001" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${script_dir}/pio.sh" run --target upload --upload-port "$1"

