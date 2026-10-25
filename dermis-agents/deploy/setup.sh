#!/usr/bin/env bash
# Run on the server, from the dermis-agents folder: bash deploy/setup.sh
# Installs the venv, checks the data and secrets are there, and starts the agents as a service.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

[ -f .agent_seed ] || [ -n "$(grep -s '^AGENT_SEED_SECRET=.' .env)" ] || {
  echo "Missing .agent_seed: copy it from your laptop, or the agents get new addresses."; exit 1; }
[ -f .env ] || echo "Note: no .env, so no ASI:One LLM and no SpacetimeDB (local dataset only)."
.venv/bin/python -c "from dermis_agents import data; eps, src = data.source().list_episodes(); print(len(eps), 'episodes from', src)"

sed -e "s|__USER__|$USER|" -e "s|__DIR__|$DIR|g" deploy/dermis-agents.service | sudo tee /etc/systemd/system/dermis-agents.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now dermis-agents
sleep 5
.venv/bin/python run.py --addresses
echo "Running. Logs: journalctl -u dermis-agents -f"
