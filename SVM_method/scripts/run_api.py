"""uvicorn plan1_app.integration.api:app --reload"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    import uvicorn

    host = "127.0.0.1"
    port = 8000
    print(f"API running at: http://{host}:{port}")
    print(f"API docs at:    http://{host}:{port}/docs")
    uvicorn.run("plan1_app.integration.api:app", host=host, port=port, reload=False)
