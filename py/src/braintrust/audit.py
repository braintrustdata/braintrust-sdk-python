"""
Utilities for working with audit headers.
"""

# PEP 810: defer these imports until first use on Python 3.15+. Ignored on older versions.
__lazy_modules__ = {"gzip"}

import base64
import gzip as _gzip
import json
from typing import TypedDict


class AuditResource(TypedDict):
    type: str
    id: str
    name: str


def parse_audit_resources(hdr: str) -> list[AuditResource]:
    j = json.loads(hdr)
    if j["v"] == 1:
        return json.loads(_gzip.decompress(base64.b64decode(j["p"])))
    else:
        raise ValueError(f"Unsupported audit resources protocol version: {j['v']}")
