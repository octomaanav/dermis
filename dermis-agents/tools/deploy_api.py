"""Update the hosted agents through the Agentverse API with AGENTVERSE_API_KEY (no avctl login, no expiry).

  python tools/deploy_api.py            rebuild, check, upload code and restart all hosted agents
  python tools/deploy_api.py hub        just these

The agents must already exist (first deploy: tools/deploy_hosted.py). Addresses, handles, secrets and
profiles stay as they are; this replaces the code and restarts each agent, then shows its self-check.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_hosted as hb  # noqa: E402
from logs import API, api_key, logs  # noqa: E402


def files_payload(folder: Path) -> str:
    files = [{"language": "python", "name": f.name, "value": f.read_text()} for f in sorted(folder.glob("*.py"))]
    env = folder / ".env"
    if env.exists():
        files.insert(0, {"language": "plaintext", "name": ".env", "value": env.read_text()})
    return json.dumps(files)


def main() -> None:
    key = api_key()
    H = {"Authorization": f"Bearer {key}"}
    keys = sys.argv[1:] or hb.HOSTED
    url = hb.URL_FILE.read_text().strip()
    hb.build(url)
    for k in keys:
        r = subprocess.run([sys.executable, str(Path(__file__).parent / "check_hosted.py"), str(hb.OUT / k)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"❌ {k} doesn't load like it will on Agentverse:\n{(r.stdout + r.stderr)[-800:]}")
    print(f"✅ {len(keys)} agents load like they will on Agentverse")
    with httpx.Client(headers=H, timeout=60) as c:
        for k in keys:
            addr = hb.deployed_address(k)
            if not addr:
                print(f"❌ {k}: not deployed yet (use tools/deploy_hosted.py first)")
                continue
            stop = c.post(f"{API}/{addr}/stop")  # code can only be replaced while the agent is stopped
            up = c.put(f"{API}/{addr}/code", json={"code": files_payload(hb.OUT / k)})
            start = c.post(f"{API}/{addr}/start")
            ok = up.status_code < 300 and start.status_code < 300
            print(f"{'✅' if ok else '❌'} {k:15} upload {up.status_code} · stop {stop.status_code} · start {start.status_code}"
                  + ("" if ok else f"  {up.text[:200]} {start.text[:200]}"))
    print("\nwaiting for startup self-checks…")
    time.sleep(30)
    for k in keys:
        entries = logs(hb.deployed_address(k), key)
        last = next((e for e in reversed(entries) if "self-check" in e["log_entry"] or "Error" in e["log_entry"]), None)
        print(f"  {k:15} {last['log_entry'][:150] if last else 'no self-check yet (check again shortly)'}")


if __name__ == "__main__":
    main()
