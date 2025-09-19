import json
from pathlib import Path

LOG_FILE = Path("logs/results.jsonl")
LOG_FILE.parent.mkdir(exist_ok=True, parents=True)

def log_result(entry: dict):
    """Append a single log entry to a daily JSONL file"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
