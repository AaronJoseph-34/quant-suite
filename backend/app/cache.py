import time
from copy import deepcopy
from typing import Any, Dict, Tuple

# key -> (timestamp, data)
_CACHE: Dict[str, Tuple[float, Any]] = {}

TTL_SECONDS = 300  # 5 minutes


def get(key: str):
    entry = _CACHE.get(key)
    if not entry:
        return None

    ts, value = entry
    if time.time() - ts > TTL_SECONDS:
        del _CACHE[key]
        return None

    return deepcopy(value)


def set(key: str, value: Any):
    _CACHE[key] = (time.time(), deepcopy(value))
