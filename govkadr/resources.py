"""Memory budgeting.

On Windows (WDDM) every CUDA allocation is backed by system *commit* one-to-one, so the usable
working set is min(free VRAM, free commit), not free VRAM alone. The pipeline sizes its networks
from this live budget instead of assuming the full 8 GB card.
"""

from __future__ import annotations

import ctypes
import gc
import sys


def free_commit_gb() -> float:
    if sys.platform == "win32":

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullAvailPageFile / 2**30
    import psutil

    return psutil.virtual_memory().available / 2**30


def free_vram_gb() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / 2**30


def gpu_budget_gb(safety_gb: float = 0.6) -> float:
    """GB that new GPU allocations may use right now without exhausting VRAM or system commit."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    vram = free_vram_gb()
    return max(0.0, min(vram, free_commit_gb() if sys.platform == "win32" else vram) - safety_gb)


def snapshot() -> dict:
    out = {"free_commit_gb": round(free_commit_gb(), 2)}
    try:
        import torch

        if torch.cuda.is_available():
            out["free_vram_gb"] = round(free_vram_gb(), 2)
            out["torch_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    except ImportError:
        pass
    return out
