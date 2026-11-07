"""How much this machine can take, so the heavy parts of Lumexa size themselves instead of getting the process killed.

The backend runs on a 4 GB Android tablet next to the CalculiX solver and a tunnel. Android's low-memory killer ends a process that grows
too much, and then the whole backend is gone. So every chunked loop, mesh limit and tessellation tolerance reads its numbers from here.
Profiles: tight (little free memory: the usual state of a 4 GB tablet), normal, roomy (a PC). Force one with LUMEXA_PROFILE=tight|normal|roomy.
"""
from __future__ import annotations

import os
import re

_PROFILES = {
    # chunk_entries: ray x triangle pairs per numpy block (about 110 bytes each); max_skin_triangles: refuse bigger skins with a clear message;
    # stl_tol: (linear mm, angular rad) used to tessellate parts for the assembly view; max_upload_mb: biggest file the backend will read
    "tight": {"chunk_entries": 200_000, "max_skin_triangles": 120_000, "wall_samples": 80, "stl_tol": (0.05, 0.20), "max_upload_mb": 30},
    "normal": {"chunk_entries": 600_000, "max_skin_triangles": 250_000, "wall_samples": 120, "stl_tol": (0.03, 0.15), "max_upload_mb": 60},
    "roomy": {"chunk_entries": 1_500_000, "max_skin_triangles": 600_000, "wall_samples": 150, "stl_tol": (0.02, 0.10), "max_upload_mb": 150},
}


def mem_available_mb(meminfo_text=None):
    """Memory the system could give a process right now, in MB, or None when it cannot be read (not Linux)."""
    try:
        txt = meminfo_text if meminfo_text is not None else open("/proc/meminfo").read()
    except OSError:
        return None
    m = re.search(r"MemAvailable:\s+(\d+)\s*kB", txt)
    if m:
        return int(m.group(1)) / 1024.0
    free = re.search(r"MemFree:\s+(\d+)\s*kB", txt)
    cached = re.search(r"^Cached:\s+(\d+)\s*kB", txt, re.M)
    return (int(free.group(1)) + (int(cached.group(1)) if cached else 0)) / 1024.0 if free else None


def mem_total_mb(meminfo_text=None):
    try:
        txt = meminfo_text if meminfo_text is not None else open("/proc/meminfo").read()
    except OSError:
        return None
    m = re.search(r"MemTotal:\s+(\d+)\s*kB", txt)
    return int(m.group(1)) / 1024.0 if m else None


def profile(avail_mb="auto", total_mb="auto"):
    """The limits to use now. An unreadable memory size is treated as tight: assume the small device."""
    forced = os.environ.get("LUMEXA_PROFILE", "").strip().lower()
    if avail_mb == "auto":
        avail_mb = mem_available_mb()
    if total_mb == "auto":
        total_mb = mem_total_mb()
    if forced in _PROFILES:
        name = forced
    elif total_mb is not None and total_mb < 6500:
        name = "tight" if (avail_mb is None or avail_mb < 2500) else "normal"        # a 4 GB tablet never gets 'roomy'
    elif avail_mb is None:
        name = "tight"
    else:
        name = "tight" if avail_mb < 1800 else ("normal" if avail_mb < 3500 else "roomy")
    out = dict(_PROFILES[name])
    out.update({"name": name, "mem_available_mb": None if avail_mb is None else round(avail_mb), "mem_total_mb": None if total_mb is None else round(total_mb)})
    return out


def stl_triangles_from_size(n_bytes):
    """Triangles in a binary STL of this size (84-byte header, 50 bytes per triangle)."""
    return max(0, (int(n_bytes) - 84) // 50)
