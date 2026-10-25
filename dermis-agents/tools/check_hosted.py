"""Load a generated hosted agent the way Agentverse does, before deploying it.

  python tools/check_hosted.py hosted/hub

Agentverse runs each module's code before registering it in sys.modules and may not set __file__,
which breaks things like dataclasses under postponed annotations. This loader reproduces that, so the
failure shows up here instead of in the hosted agent's logs.
"""

import importlib.abc
import importlib.util
import os
import sys
import types


class LateRegister(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, folder: str):
        self.folder = folder

    def find_spec(self, name, path, target=None):
        f = os.path.join(self.folder, name + ".py")
        return importlib.util.spec_from_loader(name, self) if name.startswith("dermis_") and os.path.exists(f) else None

    def create_module(self, spec):
        return types.ModuleType(spec.name)

    def exec_module(self, module):
        with open(os.path.join(self.folder, module.__name__ + ".py")) as fh:
            src = fh.read()
        sys.modules.pop(module.__name__, None)  # not registered while its code runs, no __file__
        exec(compile(src, module.__name__ + ".py", "exec"), module.__dict__)
        sys.modules[module.__name__] = module


def main() -> None:
    folder = os.path.abspath(sys.argv[1])
    os.chdir(folder)
    sys.meta_path.insert(0, LateRegister(folder))
    with open("agent.py") as fh:
        ns = {"__name__": "agent_check"}
        exec(compile(fh.read(), "agent.py", "exec"), ns)
    print(f"ok {os.path.basename(folder)}: {len(ns['agent'].protocols)} protocols")


if __name__ == "__main__":
    main()
