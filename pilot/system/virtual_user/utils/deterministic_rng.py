# virtual_user/utils/deterministic_rng.py
import hashlib
import numpy as np
import uuid


def _u64_from_key(*parts: str) -> int:
    s = "|".join(parts).encode("utf-8")
    h = hashlib.blake2b(s, digest_size=8).digest()
    return int.from_bytes(h, "big", signed=False)


def rng_from_key(*parts: str) -> np.random.Generator:
    seed = _u64_from_key(*map(str, parts))
    # PCG64 is stable and good for simulation
    return np.random.default_rng(seed)


def deterministic_uuid(*parts: str) -> str:
    seed = _u64_from_key(*map(str, parts))
    # Build a UUID from deterministic 128 bits
    h = hashlib.blake2b(str(seed).encode("utf-8"), digest_size=16).digest()
    return str(uuid.UUID(bytes=h, version=4))
