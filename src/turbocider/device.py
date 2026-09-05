"""Privacy-safe Apple Silicon device discovery and ANE profile matching."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from turbocider.paths import PACKAGE_ROOT


def _command(*argv: str) -> str:
    try:
        return subprocess.run(
            argv,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _profiler(kind: str) -> Dict[str, Any]:
    raw = _command("system_profiler", kind, "-json")
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
        values = payload.get(kind, [])
        return dict(values[0]) if values else {}
    except (ValueError, TypeError):
        return {}


@dataclass(frozen=True)
class DeviceInfo:
    architecture: str
    machine_model: str
    chip: str
    gpu_cores: Optional[int]
    memory_bytes: Optional[int]
    os_version: str
    os_build: str

    @property
    def memory_gib(self) -> Optional[float]:
        if self.memory_bytes is None:
            return None
        return self.memory_bytes / (1024 ** 3)

    @property
    def fingerprint(self) -> str:
        stable = {
            "architecture": self.architecture,
            "machine_model": self.machine_model,
            "chip": self.chip,
            "gpu_cores": self.gpu_cores,
            "memory_bytes": self.memory_bytes,
        }
        rendered = json.dumps(stable, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "machine_model": self.machine_model,
            "chip": self.chip,
            "gpu_cores": self.gpu_cores,
            "memory_bytes": self.memory_bytes,
            "memory_gib": self.memory_gib,
            "os_version": self.os_version,
            "os_build": self.os_build,
            "fingerprint": self.fingerprint,
        }


@lru_cache(maxsize=1)
def current_device() -> DeviceInfo:
    hardware = _profiler("SPHardwareDataType")
    display = _profiler("SPDisplaysDataType")
    memory_text = _command("sysctl", "-n", "hw.memsize")
    cores_text = str(display.get("sppci_cores", ""))
    return DeviceInfo(
        architecture=platform.machine(),
        machine_model=str(hardware.get("machine_model") or _command("sysctl", "-n", "hw.model")),
        chip=str(hardware.get("chip_type") or display.get("sppci_model") or ""),
        gpu_cores=int(cores_text) if cores_text.isdigit() else None,
        memory_bytes=int(memory_text) if memory_text.isdigit() else None,
        os_version=platform.mac_ver()[0],
        os_build=_command("sw_vers", "-buildVersion"),
    )


class DeviceProfileCatalog:
    def __init__(self, search_paths: Iterable[Path] = ()):
        self.search_paths = [PACKAGE_ROOT / "device-profiles"]
        self.search_paths.extend(Path(path) for path in search_paths)
        self.profiles: Dict[str, Dict[str, Any]] = {}
        for directory in self.search_paths:
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.json")):
                raw = json.loads(path.read_text(encoding="utf-8"))
                self.profiles[str(raw["id"])] = raw

    def match(
        self, profile_ids: Iterable[str], device: Optional[DeviceInfo] = None
    ) -> Dict[str, Any]:
        actual = device or current_device()
        reports = []
        for profile_id in profile_ids:
            profile = self.profiles.get(str(profile_id))
            if profile is None:
                reports.append({"id": str(profile_id), "matched": False, "failures": ["profile not installed"]})
                continue
            failures = _match_failures(actual, dict(profile.get("match", {})))
            reports.append({
                "id": str(profile_id),
                "matched": not failures,
                "failures": failures,
                "description": str(profile.get("description", "")),
            })
        matched = next((item["id"] for item in reports if item["matched"]), None)
        return {
            "device": actual.as_dict(),
            "matched": matched is not None,
            "matched_profile": matched,
            "profiles": reports,
        }


def _match_failures(device: DeviceInfo, rules: Dict[str, Any]) -> List[str]:
    failures: List[str] = []
    exact_fields = {
        "architectures": device.architecture,
        "machine_models": device.machine_model,
        "chips": device.chip,
        "gpu_cores": device.gpu_cores,
    }
    for key, actual in exact_fields.items():
        expected = rules.get(key)
        if expected is not None and actual not in expected:
            failures.append("%s=%s not in %s" % (key, actual, expected))
    memory = rules.get("memory_gib")
    if isinstance(memory, dict):
        actual_memory = device.memory_gib
        minimum = float(memory.get("min", 0.0))
        maximum = float(memory.get("max", float("inf")))
        if actual_memory is None or not minimum <= actual_memory <= maximum:
            failures.append(
                "memory_gib=%s not in %.1f..%.1f"
                % (actual_memory if actual_memory is not None else "unknown", minimum, maximum)
            )
    return failures
