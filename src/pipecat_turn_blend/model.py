"""Locates turn-blend.onnx, downloading it on first use.

The model (56 MB) is published as a release asset of this repository rather than inside the Python package.
It is downloaded once, checked against its SHA-256 and kept in the user's cache directory.
"""

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

MODEL_VERSION = "0.1.0"
MODEL_URL = f"https://github.com/adhiraj4802/pipecat-turn-blend/releases/download/v{MODEL_VERSION}/turn-blend.onnx"
MODEL_SHA256 = "f755abf26761688d22e0cd5f05ced041d5d2b155da50e54eb7330a64ef29474c"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_path(cache_dir: str | os.PathLike | None = None) -> str:
    """Return the path of turn-blend.onnx, downloading it if it is not cached yet.

    Args:
        cache_dir: Folder to keep the model in. Defaults to ``$XDG_CACHE_HOME/turn-blend`` or
            ``~/.cache/turn-blend``.

    Returns:
        Path of the verified model file.

    Raises:
        RuntimeError: If the downloaded file does not match the expected SHA-256.
    """
    base = Path(cache_dir) if cache_dir else Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "turn-blend"
    path = base / f"turn-blend-{MODEL_VERSION}.onnx"
    if path.exists():
        return str(path)
    base.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=base, suffix=".part")
    os.close(fd)
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        found = _sha256(Path(tmp))
        if found != MODEL_SHA256:
            raise RuntimeError(f"{MODEL_URL}: SHA-256 is {found}, expected {MODEL_SHA256}")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return str(path)
