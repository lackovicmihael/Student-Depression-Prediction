from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["AZUREML_MODEL_DIR"] = str(ROOT / "outputs" / "local_model")

from azure.score import init, run

init()
payload = json.loads((ROOT / "sample_request.json").read_text(encoding="utf-8"))
print(run(payload))
