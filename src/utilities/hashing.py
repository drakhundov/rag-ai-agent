import hashlib
import json


def compute_hash(data: object, *args, **kwargs) -> str:
    json_bytes = json.dumps(data, *args, **kwargs).encode("utf-8")
    return hashlib.sha256(json_bytes).hexdigest()
