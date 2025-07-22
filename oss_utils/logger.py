"""
Centralised + toggleable logging for the Decompute SDK
Only UserLogger prints to console; SDKLogger writes JSON to disk.
"""

from __future__ import annotations
import logging, json, os, sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any

# --- Helper -----------------------------------------------------------------

def _default_logs_dir() -> Path:
    """
    Platform-correct writable directory:
      •  Windows →  %LOCALAPPDATA%\\DecomputeSDK\\logs
      •  macOS   →  ~/Library/Logs/DecomputeSDK
      •  Linux   →  ~/.cache/decompute-sdk/logs
    """
    if sys.platform.startswith("win"):
        root = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "DecomputeSDK" / "logs"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "DecomputeSDK"
    else:                           # linux, wsl, etc.
        return Path.home() / ".cache" / "decompute-sdk" / "logs"


def _logs_file_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = _default_logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"sdk_{ts}.log"


# --- Structured JSON formatter ----------------------------------------------

class _StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if hasattr(record, "structured_data"):
            # Convert non-serializable objects to strings
            entry["data"] = {
                k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v 
                for k, v in record.structured_data.items()
            }

        return json.dumps(entry, ensure_ascii=False)


# --- Core SDK Logger ---------------------------------------------------------

class SDKLogger(logging.Logger):
    """
    Wrapper around ``logging.Logger`` that ALWAYS logs to a file and – **optionally**
    – also to the console if the `internal_console_logs` feature flag is enabled.
    """
    def __init__(
        self,
        name: str = "decompute_sdk",
        *,
        console: bool,
        level_file: int,
        structured: bool = True,
        log_file: Optional[Path] = None,
    ):
        super().__init__(name, level=logging.DEBUG)     # accept every level; handlers filter

        # Prevent duplicate handlers when get_logger() is called repeatedly.
        self.handlers.clear()

        # 1️⃣ File handler ----------------------------------------------------
        file_handler = logging.FileHandler(str(log_file or _logs_file_path()), encoding="utf-8")
        file_handler.setLevel(level_file)
        file_handler.setFormatter(_StructuredFormatter() if structured else
                                  logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.addHandler(file_handler)

        # 2️⃣ Optional console handler (mostly OFF in production) ------------
        if console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(level_file)
            ch.setFormatter(_StructuredFormatter() if structured else
                            logging.Formatter("%(message)s"))
            self.addHandler(ch)

    def _log_struct(self, lvl: int, msg: str, *args, **kw):
        """
        Forward to logging.Logger._log().
        *  args  → regular %-style arguments (may be empty)
        *  kw    → captured as structured data and stored under `extra`
        """
        extra = {"structured_data": kw} if kw else None
        super()._log(lvl, msg, args, extra=extra)

    # 2. CONVENIENCE METHODS  (now accept *args as normal)
    def debug   (self, m, *a, **k): self._log_struct(logging.DEBUG,    m, *a, **k)
    def info    (self, m, *a, **k): self._log_struct(logging.INFO,     m, *a, **k)
    def warning (self, m, *a, **k): self._log_struct(logging.WARNING,  m, *a, **k)
    def error   (self, m, *a, **k): self._log_struct(logging.ERROR,    m, *a, **k)
    def critical(self, m, *a, **k): self._log_struct(logging.CRITICAL, m, *a, **k)


# --- Public factory ----------------------------------------------------------

_logger_singleton: Optional[SDKLogger] = None
def get_logger(
     name: str = "decompute_sdk",
     *,                                   # everything after this remains keyword-only
     force_console: Optional[bool] = None,
     structured: bool = True,
 ) -> SDKLogger:
    """
    Creates (or returns) the singleton SDKLogger.
    The `force_console` param overrides feature-flags (handy for unit tests).
    """
    from .feature_flags import is_feature_enabled  # local import, avoids cycle

    global _logger_singleton
    if _logger_singleton is not None:
        return _logger_singleton

    console_enabled_flag = is_feature_enabled("internal_console_logs")
    verbose_flag         = is_feature_enabled("verbose_internal_logs")
    console_final        = force_console if force_console is not None else console_enabled_flag
    file_level           = logging.DEBUG if verbose_flag else logging.INFO

    _logger_singleton = SDKLogger(
        name=name,
        console=console_final,
        level_file=file_level,
        structured=structured,
    )
    return _logger_singleton


# --- Utility: log maintenance ------------------------------------------------

def clear_logs(force: bool = False, keep_latest: int = 3) -> int:
    """
    Delete old log files.
    • If `force` is True, *all* files are removed.
    • Otherwise, keep the `keep_latest` most recent ones.
    Returns the number of files deleted.
    """
    folder = _default_logs_dir()
    if not folder.exists():
        return 0

    files = sorted(folder.glob("sdk_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    killers = files if force else files[keep_latest:]
    for f in killers:
        try:
            f.unlink()
        except Exception:
            pass
    return len(killers)
