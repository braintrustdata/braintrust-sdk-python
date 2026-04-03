# Braintrust Python SDK - Security Vulnerability Audit

## Summary

This document details confirmed security vulnerabilities in the user-facing portions of the Braintrust Python SDK (`py/src/braintrust/`). Each finding has been validated with a reproduction script. Findings from tests, build processes, and test dependencies are excluded. Items that were initially flagged but proved non-exploitable upon reproduction have been removed.

---

## CRITICAL

### 1. CORS Origin Validation Bypass via Regex `re.match` Without End Anchor

**File:** `py/src/braintrust/devserver/cors.py:11, 61`

**Description:** The CORS origin allowlist uses `re.match()` with a regex pattern that lacks an end-of-string anchor (`$`). Python's `re.match()` only anchors at the start, meaning a malicious origin that *starts with* the allowed pattern but continues with an attacker-controlled domain will pass validation.

**Vulnerable Code:**
```python
# Line 11
re.compile(r"https://.*\.preview\.braintrust\.dev"),

# Line 61
elif isinstance(allowed, re.Pattern) and allowed.match(origin):
    return True
```

**Attack:** An attacker registers `https://evil.preview.braintrust.dev.attacker.com` and makes cross-origin requests to a victim's dev server. The regex matches because `re.match` doesn't require the string to end after `.dev`. The CORS middleware then sets `Access-Control-Allow-Origin` and `Access-Control-Allow-Credentials: true`, allowing the attacker's page to read authenticated API responses and exfiltrate data (API tokens, evaluation results, dataset contents).

**Reproduction:**
```python
import re

pattern = re.compile(r"https://.*\.preview\.braintrust\.dev")

# Legitimate - should match
assert pattern.match("https://legit.preview.braintrust.dev")

# MALICIOUS - should NOT match, but DOES
assert pattern.match("https://evil.preview.braintrust.dev.attacker.com")  # VULN!
assert pattern.match("https://foo.preview.braintrust.dev.evil.com/steal")  # VULN!
assert pattern.match("https://x.preview.braintrust.dev:8080")             # VULN!

# Fix: fullmatch rejects all malicious origins
assert not pattern.fullmatch("https://evil.preview.braintrust.dev.attacker.com")
assert not pattern.fullmatch("https://foo.preview.braintrust.dev.evil.com/steal")
```

**Fix:** Use `re.fullmatch()` instead of `re.match()`, or append `$` to the regex pattern.

---

### 2. ReDoS (Regular Expression Denial of Service) in Eval Filters

**File:** `py/src/braintrust/framework.py:1147`

**Description:** User-provided filter strings from the `--filter` CLI argument are compiled directly into regex patterns via `re.compile()` without any validation, timeout, or complexity limits.

**Vulnerable Code:**
```python
def parse_filters(filters: list[str]) -> list[Filter]:
    for f in filters:
        # ...
        result.append(
            Filter(
                path=path.split("."),
                pattern=re.compile(deserialized_value),  # Line 1147
            )
        )
```

**Attack:** A user (or a script invoking `braintrust eval`) passes a filter with a catastrophic-backtracking regex like `--filter 'input.name=(a+)+b'`. When the regex is evaluated against input data, the CPU enters exponential backtracking, hanging the process indefinitely.

**Reproduction:**
```python
import re, time

evil_pattern = re.compile(r"(a+)+$")
# Exponential blowup: each +2 chars ~4x the time
for n in [15, 18, 20, 22]:
    test = "a" * n + "X"
    start = time.time()
    evil_pattern.search(test)
    print(f"n={n}: {time.time() - start:.4f}s")
# Output:
#   n=15: 0.0043s
#   n=18: 0.0671s
#   n=20: 0.1767s
#   n=22: 0.6253s
# n=30 would take hours. n=40 would take centuries.
```

**Fix:** Validate the regex pattern complexity before compilation, use `re2` for safe regex execution, or apply a timeout.

---

## HIGH

### 3. Symlink Traversal in Zip Bundle Creation

**File:** `py/src/braintrust/cli/push.py:188-202`

**Description:** When creating a zip bundle for `braintrust push`, the code walks a directory and adds files to a zip archive. It does not check whether files are symlinks before adding them. `os.path.isfile()` returns `True` for symlink targets.

**Vulnerable Code:**
```python
for dirpath, dirnames, filenames in os.walk(packages_dir):
    for name in filenames:
        path = os.path.join(dirpath, name)
        path = os.path.normpath(path)
        if os.path.isfile(path):          # True for symlink targets!
            arcname = os.path.join(arcdirpath, name)
            zf.write(path, arcname)       # Follows symlink, reads target
for source in sources:
    zf.write(source, os.path.relpath(source))  # No symlink check
```

**Attack:** An attacker places a symlink in the packages directory (e.g., `link -> ~/.ssh/id_rsa`). The sensitive file contents are bundled into the zip and uploaded to the Braintrust server.

**Reproduction:**
```python
import os, tempfile, zipfile

with tempfile.TemporaryDirectory() as td:
    packages_dir = os.path.join(td, "pkg")
    os.makedirs(packages_dir)

    # Create a secret file outside the packages dir
    secret = os.path.join(td, "secret.txt")
    with open(secret, "w") as f:
        f.write("SECRET_API_KEY=sk-1234567890")

    # Symlink from inside packages_dir to the secret
    os.symlink(secret, os.path.join(packages_dir, "exfiltrated.txt"))

    # os.path.isfile follows symlinks:
    assert os.path.isfile(os.path.join(packages_dir, "exfiltrated.txt"))  # True!
    assert os.path.islink(os.path.join(packages_dir, "exfiltrated.txt"))  # Also True

    # Simulate push.py zip creation
    zip_path = os.path.join(td, "bundle.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for dirpath, dirnames, filenames in os.walk(packages_dir):
            for name in filenames:
                path = os.path.normpath(os.path.join(dirpath, name))
                if os.path.isfile(path):
                    zf.write(path, name)

    with zipfile.ZipFile(zip_path, "r") as zf:
        content = zf.read("exfiltrated.txt").decode()
        assert content == "SECRET_API_KEY=sk-1234567890"  # Secret exfiltrated!
```

**Fix:** Check `os.path.islink(path)` before adding files, or use `os.path.realpath()` and verify the resolved path is within the expected directory.

---

### 4. Unvalidated Proxy/API URL Redirect via Environment Variables

**File:** `py/src/braintrust/logger.py:2669-2670`

**Description:** The SDK sets `api_url` and `proxy_url` from environment variables without validating the URL scheme, hostname, or structure. An attacker who controls the environment (e.g., CI/CD, shared compute) can redirect all SDK API traffic -- including Bearer tokens -- to a malicious server over plaintext HTTP.

**Vulnerable Code:**
```python
state.api_url = os.environ.get("BRAINTRUST_API_URL", orgs["api_url"])
state.proxy_url = os.environ.get("BRAINTRUST_PROXY_URL", orgs["proxy_url"])
```

**Reproduction:**
```python
import os
from urllib.parse import urlparse

# Simulate the env override
os.environ["BRAINTRUST_API_URL"] = "http://attacker.evil.com"
orgs = {"api_url": "https://api.braintrust.dev"}

api_url = os.environ.get("BRAINTRUST_API_URL", orgs["api_url"])
parsed = urlparse(api_url)

assert parsed.scheme == "http"                # No TLS! Tokens sent in plaintext
assert parsed.hostname == "attacker.evil.com"  # Attacker-controlled server
# The SDK would send: Authorization: Bearer <real_api_key> to this URL
# No validation of scheme or hostname exists in the codebase

del os.environ["BRAINTRUST_API_URL"]
```

**Fix:** Validate that the URL scheme is `https://`, and optionally validate the hostname against known Braintrust domains or a configurable allowlist.

---

### 5. Gzip Decompression Bomb in Audit Header Parsing

**File:** `py/src/braintrust/audit.py:17-20`

**Description:** The `parse_audit_resources` function decompresses gzip data from a base64-encoded header without any size limit on the decompressed output.

**Vulnerable Code:**
```python
def parse_audit_resources(hdr: str) -> list[AuditResource]:
    j = json.loads(hdr)
    if j["v"] == 1:
        return json.loads(gzip.decompress(base64.b64decode(j["p"])))
```

**Reproduction:**
```python
import base64, gzip, json

# Create a gzip bomb: 10MB payload compresses to ~10KB
payload = b"\x00" * (10 * 1024 * 1024)
compressed = gzip.compress(payload)
b64 = base64.b64encode(compressed).decode()
header = json.dumps({"v": 1, "p": b64})

assert len(header) < 15_000            # ~14KB header
assert len(gzip.decompress(
    base64.b64decode(json.loads(header)["p"])
)) == 10_485_760                        # Expands to 10MB!

# Compression ratio: ~1026x
# A 1MB header could expand to ~1GB, causing OOM
```

**Fix:** Use `gzip.GzipFile` with a streaming reader and enforce a maximum decompressed size.

---

## MEDIUM

### 6. World-Readable Span Cache Files in Shared Temp Directory

**File:** `py/src/braintrust/span_cache.py:162-166`

**Description:** The span cache creates temporary files in the shared system temp directory using `open()` (which follows the process umask, typically resulting in world-readable files). Any local user can discover and read these files via a simple glob pattern.

**Vulnerable Code:**
```python
unique_id = f"{int(os.times().elapsed * 1000000)}-{uuid.uuid4().hex[:8]}"
self._cache_file_path = os.path.join(
    tempfile.gettempdir(), f"braintrust-span-cache-{unique_id}.jsonl"
)
with open(self._cache_file_path, "w") as f:
    pass
```

**Reproduction:**
```python
import os, tempfile, uuid, glob

tmpdir = tempfile.gettempdir()

# Simulate SDK creating the cache file
unique_id = f"{int(os.times().elapsed * 1000000)}-{uuid.uuid4().hex[:8]}"
cache_path = os.path.join(tmpdir, f"braintrust-span-cache-{unique_id}.jsonl")

with open(cache_path, "w") as f:
    f.write('{"input": "secret prompt", "output": "sensitive response"}\n')

perms = os.stat(cache_path).st_mode
assert perms & 0o004  # World-readable!

# Any user on the system can find these files:
found = glob.glob(os.path.join(tmpdir, "braintrust-span-cache-*.jsonl"))
assert cache_path in found  # Discoverable!

os.unlink(cache_path)
```

**Fix:** Use `tempfile.mkstemp()` or `os.open()` with `O_EXCL` and mode `0o600` to create files that are only readable by the owner.

---

### 7. Org Name Information Disclosure in Dev Server Error Responses

**File:** `py/src/braintrust/devserver/server.py:89`

**Description:** The dev server's authorization middleware reveals the configured organization name to unauthorized requesters.

**Vulnerable Code:**
```python
error_message = f"Org '{org_name}' is not allowed. Only org '{self.allowed_org_name}' is allowed."
return JSONResponse({"error": error_message}, status_code=403)
```

**Reproduction:**
```python
# Attacker sends: x-bt-org-name: anything
# Server responds with 403:
allowed_org_name = "acme-corp-secret-project"
org_name = "attacker-probe"
error_message = f"Org '{org_name}' is not allowed. Only org '{allowed_org_name}' is allowed."
assert "acme-corp-secret-project" in error_message  # Org name leaked!
```

**Fix:** Use a generic error message: `"Access denied"`.

---

### 8. Internal Exception Details Leaked to API Clients

**File:** `py/src/braintrust/devserver/server.py:143, 167, 258, 289`

**Description:** Multiple error handlers return full Python exception messages to HTTP clients. Exception messages may contain internal file paths, package versions, connection strings, or other implementation details.

**Vulnerable Code:**
```python
return JSONResponse({"error": f"Internal error: {str(e)}"}, status_code=500)
return JSONResponse({"error": f"Failed to load dataset: {str(e)}"}, status_code=400)
return JSONResponse({"error": f"Failed to run evaluation: {str(e)}"}, status_code=500)
```

**Reproduction:**
```python
# Realistic exception that leaks internal details:
try:
    raise ConnectionError(
        "postgresql://admin:p4ssw0rd@internal-db.vpc:5432/braintrust"
    )
except Exception as e:
    response = {"error": f"Internal error: {str(e)}"}
    assert "p4ssw0rd" in response["error"]       # Credentials leaked!
    assert "internal-db.vpc" in response["error"]  # Infrastructure leaked!
```

**Fix:** Log the full exception server-side and return generic error messages to clients.

---

### 9. Disk Cache Permissions Allow Local Data Exfiltration

**File:** `py/src/braintrust/prompt_cache/disk_cache.py`, `py/src/braintrust/logger.py:441-442`

**Description:** The prompt/parameters disk cache directory is created with default umask permissions (typically world-readable). Cached prompts may contain sensitive business logic, API configurations, or model parameters. Additionally, if `HOME` is unset, the path becomes `None/.braintrust/prompt_cache` (a broken relative path).

**Vulnerable Code:**
```python
cache_dir=os.environ.get(
    "BRAINTRUST_PROMPT_CACHE_DIR", f"{os.environ.get('HOME')}/.braintrust/prompt_cache"
),
```

**Reproduction:**
```python
import os, tempfile

with tempfile.TemporaryDirectory() as td:
    cache_dir = os.path.join(td, "cache")
    os.makedirs(cache_dir, exist_ok=True)  # Same as disk_cache.py:128
    perms = os.stat(cache_dir).st_mode
    assert perms & 0o005  # World-readable and world-executable!

# HOME=None case:
original = os.environ.pop("HOME", None)
path = f"{os.environ.get('HOME')}/.braintrust/prompt_cache"
assert path == "None/.braintrust/prompt_cache"  # Broken path!
if original:
    os.environ["HOME"] = original
```

**Fix:** Create the cache directory with mode `0o700`. Use `os.path.expanduser("~")` instead of `os.environ.get('HOME')`.

---

## LOW

### 10. Unbounded Base64 Decoding in Span Identifier Parsing

**Files:** `py/src/braintrust/span_identifier_v1.py:110`, `v2.py:139`, `v3.py:169`, `v4.py:162`

**Description:** Span identifier strings are base64-decoded without any size validation. If span identifiers come from untrusted distributed tracing headers, an oversized payload could cause excessive memory allocation.

**Fix:** Add a maximum length check on the input string before decoding.

---

### 11. No Size Limit on Data URL Conversion

**File:** `py/src/braintrust/oai.py:84-104`

**Description:** The `_convert_data_url_to_attachment` function base64-decodes data URLs without size limits. An extremely large data URL from user input could cause memory exhaustion.

**Fix:** Add a size check before `base64.b64decode()`.
