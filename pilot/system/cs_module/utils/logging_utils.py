import json
import pprint
import numpy as np
import logging
import os


def convert_ndarrays_to_lists(obj):
    """Recursively convert all ndarrays in a structure to lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_ndarrays_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays_to_lists(item) for item in obj]
    return obj


def _safe_serialize(obj):
    # If available in this scope, reuse your ndarray converter; otherwise no-op.
    try:
        return convert_ndarrays_to_lists(obj)
    except Exception:
        return obj


def pretty(obj, *, max_chars=None, indent=2, sort_keys=True):
    """Return a multi-line, human-readable string of obj for logs."""
    try:
        s = json.dumps(_safe_serialize(obj), indent=indent, sort_keys=sort_keys, ensure_ascii=False, default=str)
    except Exception:
        s = pprint.pformat(obj, width=120, compact=False, sort_dicts=True)
    if max_chars is not None and len(s) > max_chars:
        return s[:max_chars] + f"\n... <truncated {len(s) - max_chars} chars>"
    return s


def get_logger(obj_or_name=None):
    """
    Create a module/class-scoped logger with:
    - level from LOG_LEVEL env (default INFO)
    - single StreamHandler, consistent format
    """
    if isinstance(obj_or_name, str):
        name = obj_or_name
    else:
        mod = getattr(obj_or_name, "__module__", __name__)
        cls = getattr(getattr(obj_or_name, "__class__", object), "__name__", "Logger")
        name = f"{mod}.{cls}"

    logger = logging.getLogger(name)
    logger.propagate = True  # ✅ use root’s handler/formatter

    return logger


def ctx(user_id=None, mission_id=None, pid=None, rec_id=None, extra=""):
    bits = []
    if user_id is not None:
        bits.append(f"user={user_id}")
    if mission_id is not None:
        bits.append(f"mission={mission_id}")
    if pid is not None:
        bits.append(f"pid={pid}")
    if rec_id is not None:
        bits.append(f"rec={rec_id}")
    if extra:
        bits.append(str(extra))
    return " ".join(bits)
