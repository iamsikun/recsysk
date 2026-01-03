import os
import sys
import importlib.util

# --- Add src/ to Python path (project uses src/winners_curse) ---
here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(here, ".."))
src_path = os.path.join(project_root, "src")

if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- Try to enable autoreload if in IPython ---
try:
    from IPython import get_ipython  # type: ignore
except Exception:
    get_ipython = None  # type: ignore

try:
    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic("load_ext", "autoreload")
        ip.run_line_magic("autoreload", "2")
        print("🔁 Autoreload is ON (IPython detected).")
except NameError:
    # Not in IPython (e.g., VS Code terminal or plain Python)
    print("ℹ️ Not running inside IPython — autoreload disabled.")

# --- Confirm import path ---
spec = importlib.util.find_spec("recsys")
if spec and getattr(spec, "submodule_search_locations", None):
    pkg_dir = list(spec.submodule_search_locations)[0]
    print(f"✅ Using recsys from: {pkg_dir}")
else:
    print("❌ Could not import recsys. Ensure src/recsys exists or install with 'pip install -e .'")