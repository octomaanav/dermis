"""Runtime configuration, all from environment variables (see .env.example)."""

import os
from pathlib import Path

# Hosted Agentverse agents may load modules without __file__; there the project root is the cwd.
_here = globals().get("__file__")
ROOT = Path(_here).resolve().parent.parent if _here else Path.cwd()


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# Local copy of the Dermis sim dataset. Used when SpacetimeDB is not configured or unreachable.
DATA_DIR = Path(
    os.getenv("DERMIS_DATA_DIR", ROOT.parent / "dermis-teleop" / "dataset" / "dermis_sim_v1")
)

# SpacetimeDB (HTTP SQL API). Leave SPACETIME_DB empty to use the local dataset only.
SPACETIME_HOST = os.getenv("SPACETIME_HOST", "https://maincloud.spacetimedb.com").rstrip("/")
SPACETIME_DB = os.getenv("SPACETIME_DB", "")
SPACETIME_TOKEN = os.getenv("SPACETIME_TOKEN", "")
SPACETIME_EPISODE_TABLE = os.getenv("SPACETIME_EPISODE_TABLE", "episode")
SPACETIME_FRAME_TABLE = os.getenv("SPACETIME_FRAME_TABLE", "frame")

# ASI:One LLM API (OpenAI-compatible). Without a key, agents use rule-based parsing and summaries.
ASI1_API_KEY = os.getenv("ASI1_API_KEY", "")
ASI1_BASE_URL = os.getenv("ASI1_BASE_URL", "https://api.asi1.ai/v1").rstrip("/")
ASI1_MODEL = os.getenv("ASI1_MODEL", "asi1-mini")
# The model that runs the agents' tool-calling conversations (asi1 is Fetch's agentic default).
ASI1_AGENT_MODEL = os.getenv("ASI1_AGENT_MODEL", "asi1")

# Agent identity. Seeds derive the agents' addresses, so keep AGENT_SEED_SECRET private and stable.
AGENT_SEED_SECRET = os.getenv("AGENT_SEED_SECRET", "")
MAILBOX = os.getenv("DERMIS_MAILBOX", "0") == "1"
BUREAU_PORT = int(os.getenv("DERMIS_PORT", "8000"))

# Charts are rendered by QuickChart from a URL that carries the (downsampled) series.
CHARTS = os.getenv("DERMIS_CHARTS", "1") == "1"
QUICKCHART_URL = os.getenv("QUICKCHART_URL", "https://quickchart.io/chart")

# Precomputed analysis every agent reads (built by `python -m dermis_agents.snapshot`). Hosted agents
# have no local file and fetch SNAPSHOT_URL instead.
SNAPSHOT_PATH = Path(os.getenv("DERMIS_SNAPSHOT_PATH", ROOT / "snapshot" / "dermis_snapshot.json"))
SNAPSHOT_URL = os.getenv("DERMIS_SNAPSHOT_URL", "")


def asi1() -> tuple[str, str]:
    """(api key, base url), read when called: hosted agents may get env vars after this module loads.

    DERMIS_ASI1_API_KEY (an Agentverse secret holding our own ASI:One key) wins over the ASI1_API_KEY that
    Agentverse injects, which goes through Agentverse's own LLM budget and stops when that runs out."""
    own = os.getenv("DERMIS_ASI1_API_KEY")
    if own:
        return own, (os.getenv("DERMIS_ASI1_BASE_URL") or "https://api.asi1.ai/v1").rstrip("/")
    return (os.getenv("ASI1_API_KEY") or ASI1_API_KEY,
            (os.getenv("ASI1_BASE_URL") or ASI1_BASE_URL).rstrip("/"))


def snapshot_url() -> str:
    return os.getenv("DERMIS_SNAPSHOT_URL") or SNAPSHOT_URL
