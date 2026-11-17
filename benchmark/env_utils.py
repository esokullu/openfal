"""Zero-dependency `.env` loader & environment helpers for the openfal benchmark suite."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def parse_dotenv_file(path: Path) -> Dict[str, str]:
    """Parse `KEY=VALUE` pairs from a `.env` file without mutating `os.environ`."""
    parsed: Dict[str, str] = {}
    if not path.is_file():
        return parsed
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        val = val.strip()
        if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
            val = val[1:-1]
        if key:
            parsed[key] = val
    return parsed


def load_dotenv(path: Path | None = None, override: bool = False) -> Dict[str, str]:
    """Load `benchmark/.env` (or explicit path) into `os.environ` cleanly."""
    target = path or (Path(__file__).resolve().parent / ".env")
    values = parse_dotenv_file(target)
    for k, v in values.items():
        if override or k not in os.environ or not os.environ[k]:
            os.environ[k] = v
    return values


def env_bool(name: str, default: bool = False) -> bool:
    """Interpret boolean-like environment flag consistently."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def env_csv(name: str, default: str) -> list[str]:
    """Parse comma-separated values from an environment variable."""
    raw = os.environ.get(name, default)
    return [part.strip() for part in raw.split(",") if part.strip()]
