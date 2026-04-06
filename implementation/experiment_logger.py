import json
import os
from datetime import datetime


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
LOG_PATH = os.path.join(PROJECT_ROOT, "EXPERIMENT_LOG.md")


def _format_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def append_experiment_log(title, fields):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"## {title}", f"- Date: {timestamp}"]
    for key, value in fields.items():
        lines.append(f"- {key}: {_format_value(value)}")
    lines.append("")

    with open(LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def append_json_snapshot(title, payload):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"## {title}",
        f"- Date: {timestamp}",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
    ]
    with open(LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
