"""doc_types.yaml drives the default search scope and the search_docs tool description;
prompts.yaml holds the description's wording."""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["CONFIG_DIR"] = str(ROOT / "examples" / "bdc")  # read once when config is imported below

from r_doc_mcp import config
from r_doc_mcp.api import default_types
from r_doc_mcp.mcp_server import tool_description

cfg = config.doc_types()
assert cfg["project"] == "NHLBI BioData Catalyst®"
assert list(cfg["types"]) == ["docs", "page", "faq", "video", "fellow", "update", "event"]
assert default_types() == ["docs", "page", "faq", "video"]
desc = tool_description()
for name, t in cfg["types"].items():
    assert f"- {name}: {t['description']}" in desc, name
assert "only docs, page, faq, video are searched" in desc
assert desc.startswith("Search the NHLBI BioData Catalyst® documentation database.")
assert f"(default {config.SEARCH_K})" in desc
assert '"picsure" finds "PIC-SURE"' in desc, "BDC wording comes from examples/bdc/prompts.yaml"
assert "{" not in desc, "every placeholder filled"

config.doc_types.cache_clear()
config.prompts.cache_clear()
config.CONFIG_DIR = str(ROOT / "config")  # CONFIG_DIR is read once at import; patch the module-level name (the test convention)
assert default_types() == ["docs"], "template declares docs as the only default"
assert "- docs:" in tool_description()
assert "PIC-SURE" not in tool_description(), "template wording, not BDC's"

config.doc_types.cache_clear()
empty = ROOT / "tests" / "_empty_config"
empty.mkdir(exist_ok=True)
(empty / "doc_types.yaml").write_text("project: Empty\n", encoding="utf-8")
shutil.copy(ROOT / "config" / "prompts.yaml", empty)
try:
    config.CONFIG_DIR = str(empty)
    assert default_types() == [], "no types => no default scope => unfiltered search"
    assert "every type is searched" in tool_description()
finally:
    shutil.rmtree(empty)
    config.doc_types.cache_clear()
    config.prompts.cache_clear()
    config.CONFIG_DIR = str(ROOT / "examples" / "bdc")

# each CONFIG_DIR gets its own collection in the shared DB, so examples never overwrite each other's chunks
import subprocess
out = subprocess.run([sys.executable, "-c", "from r_doc_mcp import config; print(config.COLLECTION_NAME)"],
                     capture_output=True, text=True, cwd=ROOT,
                     env={**os.environ, "CONFIG_DIR": "examples/bdc/", "COLLECTION_NAME": ""})  # empty = unset, and beats .env
assert out.stdout.strip() == "bdc", out.stdout + out.stderr

print("doc_types self-check passed")
