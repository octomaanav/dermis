"""Build and deploy the hosted agents to Agentverse with avctl, specialists first, then the hub.

  avctl auth login                                  once (browser)
  python tools/deploy_hosted.py                      build + deploy all 8
  python tools/deploy_hosted.py visualizer hub       just these (hub is always redeployed last)

The snapshot URL is the one given to build_hosted.py (saved in hosted/snapshot_url.txt).
Re-running is safe: each agent's address is kept in hosted/<key>/.avctl/config.toml, so a redeploy
updates the same agent.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_hosted as hb  # noqa: E402

AVCTL = shutil.which("avctl") or str(Path.home() / ".local" / "bin" / "avctl")


def avctl(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([AVCTL, *args], cwd=cwd, capture_output=True, text=True, stdin=subprocess.DEVNULL)


def check_snapshot(url: str) -> None:
    try:
        r = httpx.get(url, timeout=60, follow_redirects=True)
        r.raise_for_status()
        snap = r.json()
    except (httpx.HTTPError, ValueError) as e:
        sys.exit(f"Snapshot URL doesn't work ({e}). Hosted agents download it on every message, so fix this first:\n  {url}")
    if snap.get("format") != "dermis-snapshot-v1":
        sys.exit(f"{url} isn't a Dermis snapshot.")
    print(f"✅ snapshot reachable · {len(snap['episodes'])} episodes · built {snap.get('built_at')}")


def deploy(key: str) -> bool:
    name, _, _ = hb.agents.identity_of(key)
    d = hb.OUT / key
    r = avctl("hosting", "deploy", "-n", name, "--handle", hb.HANDLES[key], cwd=d)
    out = (r.stdout + r.stderr).strip()
    address = hb.deployed_address(key)
    ok = r.returncode == 0 and address.startswith("agent1")
    print(f"{'✅' if ok else '❌'} {key:15} {address or ''}")
    if not ok:
        print("   " + "\n   ".join(out.splitlines()[-8:]))
        if "limit" in out.lower():
            print("   Your Agentverse account is at its agent limit (8 on the free plan). Remove the old mailbox "
                  "agents first: agentverse.ai → My Agents → select them → Delete.")
    return ok


def main() -> None:
    if not Path(AVCTL).exists():
        sys.exit("avctl not found. Install: curl -fsSL https://github.com/fetchai/avctl/releases/latest/download/install.sh | sh")
    status = avctl("auth", "status")
    if "logged-in" not in (status.stdout + status.stderr).lower() or "logged-out" in (status.stdout + status.stderr).lower():
        sys.exit("Not logged in to Agentverse. Run:  avctl auth login")
    if not hb.URL_FILE.exists():
        sys.exit("Run tools/build_hosted.py --snapshot-url <public url> once first.")
    url = hb.URL_FILE.read_text().strip()
    check_snapshot(url)

    keys = sys.argv[1:] or hb.HOSTED
    unknown = set(keys) - set(hb.HOSTED)
    if unknown:
        sys.exit(f"Not hosted agents: {', '.join(sorted(unknown))}. Hosted: {', '.join(hb.HOSTED)}")
    specialists = [k for k in hb.HOSTED if k in keys and k != "hub"]

    hb.build(url)
    # Load every agent the way Agentverse will before uploading anything.
    broken = []
    for k in hb.HOSTED:
        r = subprocess.run([sys.executable, str(Path(__file__).parent / "check_hosted.py"), str(hb.OUT / k)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            broken.append(k)
            print(f"❌ {k} doesn't load:\n   " + "\n   ".join((r.stdout + r.stderr).strip().splitlines()[-4:]))
    if broken:
        sys.exit("Not deploying: fix the load errors above first.")
    print(f"✅ all {len(hb.HOSTED)} agents load like they will on Agentverse")
    failed = [k for k in specialists if not deploy(k)]
    # The hub learns the specialists' Agentverse addresses only after they exist, so it goes last.
    if "hub" in keys or specialists:
        hb.build(url)
        if not deploy("hub"):
            failed.append("hub")

    print()
    for k in hb.HOSTED:
        a = hb.deployed_address(k)
        if a:
            print(f"  {k:15} https://agentverse.ai/agents/details/{a}")
    if failed:
        sys.exit(f"\n{len(failed)} failed: {', '.join(failed)}. Fix the error above and re-run for just those.")
    print("\nAll deployed. Check one with:  cd hosted/hub && avctl hosting logs")


if __name__ == "__main__":
    main()
