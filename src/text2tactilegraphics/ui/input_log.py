"""Private append-only submission records, independent of the GPU queue."""

import fcntl
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def _serialize(value):
    if isinstance(value, Image.Image):
        # Do not serialize upload paths or image bytes into the preference log.
        return {"width": value.width, "height": value.height, "mode": value.mode}
    raise TypeError("Unsupported input log value")


def record_input(action: str, fields: dict) -> None:
    path = Path(
        os.getenv("TEXT2TACTILEGRAPHICS_INPUT_LOG", "private/user-inputs.jsonl")
    )
    record = {
        "submission_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "inputs": fields,
    }
    try:
        line = json.dumps(record, ensure_ascii=True, default=_serialize) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as log:
            fcntl.flock(log.fileno(), fcntl.LOCK_EX)
            log.write(line)
            log.flush()
            os.fsync(log.fileno())
    except Exception:
        # Do not run an unlogged submission or expose file paths/input in errors.
        raise ValueError("Could not save your submission. Please try again.") from None
