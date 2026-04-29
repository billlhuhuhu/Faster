import argparse
import glob
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_rapl_packages() -> Dict[str, Tuple[int, Optional[int]]]:
    """Read top-level Intel RAPL package counters in microjoules."""
    out: Dict[str, Tuple[int, Optional[int]]] = {}
    for path in glob.glob("/sys/class/powercap/intel-rapl:[0-9]/energy_uj"):
        energy_path = Path(path)
        try:
            energy = int(energy_path.read_text().strip())
        except (OSError, ValueError):
            continue
        max_path = energy_path.parent / "max_energy_range_uj"
        max_energy = None
        try:
            max_energy = int(max_path.read_text().strip())
        except (OSError, ValueError):
            pass
        out[str(energy_path.parent)] = (energy, max_energy)
    return out


def rapl_delta_uj(before: Dict[str, Tuple[int, Optional[int]]], after: Dict[str, Tuple[int, Optional[int]]]) -> int:
    total = 0
    for key, (start, max_energy) in before.items():
        if key not in after:
            continue
        end = after[key][0]
        if end >= start:
            total += end - start
        elif max_energy:
            total += (max_energy - start) + end
    return total


def parse_visible_gpu_indices() -> Optional[set]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return None
    indices = set()
    for item in visible.split(","):
        item = item.strip()
        if item.isdigit():
            indices.add(int(item))
    return indices or None


def query_gpu_power_watts(visible_indices: Optional[set]) -> Optional[float]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    total = 0.0
    count = 0
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
            watts = float(parts[1])
        except ValueError:
            continue
        if visible_indices is not None and idx not in visible_indices:
            continue
        total += watts
        count += 1
    if count == 0:
        return None
    return total


class NvidiaSmiSampler:
    def __init__(self, interval: float = 1.0):
        self.interval = max(float(interval), 0.1)
        self.visible_indices = parse_visible_gpu_indices()
        self.samples: List[Tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            watts = query_gpu_power_watts(self.visible_indices)
            if watts is not None:
                self.samples.append((time.time(), watts))
            self._stop.wait(self.interval)

    def energy_wh(self, wall_seconds: float) -> Optional[float]:
        if not self.samples:
            return None
        if len(self.samples) == 1:
            return self.samples[0][1] * wall_seconds / 3600.0
        energy_ws = 0.0
        for (t0, w0), (t1, w1) in zip(self.samples[:-1], self.samples[1:]):
            energy_ws += 0.5 * (w0 + w1) * max(t1 - t0, 0.0)
        tail = max(time.time() - self.samples[-1][0], 0.0)
        energy_ws += self.samples[-1][1] * min(tail, self.interval)
        return energy_ws / 3600.0


class ZeusWindow:
    def __init__(self, label: str):
        self.label = label
        self.monitor = None
        self.enabled = False
        self.error = None

    def __enter__(self):
        try:
            from zeus.monitor import ZeusMonitor  # type: ignore

            self.monitor = ZeusMonitor(gpu_indices=None)
            self.monitor.begin_window(self.label)
            self.enabled = True
        except Exception as exc:  # pragma: no cover - depends on optional runtime package.
            self.error = repr(exc)
            self.enabled = False
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def end(self) -> Tuple[Optional[float], Optional[str]]:
        if not self.enabled or self.monitor is None:
            return None, self.error
        try:
            measurement = self.monitor.end_window(self.label)
        except Exception as exc:  # pragma: no cover
            return None, repr(exc)

        for attr in ("total_energy", "total_energy_joules", "energy"):
            value = getattr(measurement, attr, None)
            if value is None:
                continue
            try:
                if isinstance(value, dict):
                    return sum(float(v) for v in value.values()) / 3600.0, None
                return float(value) / 3600.0, None
            except (TypeError, ValueError):
                continue
        return None, "Zeus measurement did not expose a recognized energy field."


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a command and measure wall time plus GPU/CPU energy.")
    parser.add_argument("--label", type=str, default="command")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--working_dir", type=str, default=None)
    parser.add_argument("--gpu_sampler_interval", type=float, default=1.0)
    parser.add_argument("--prefer_zeus", action="store_true", default=False)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("No command provided after --")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rapl_before = read_rapl_packages()
    sampler = NvidiaSmiSampler(interval=args.gpu_sampler_interval)
    zeus = ZeusWindow(args.label) if args.prefer_zeus else None

    start = time.time()
    sampler.start()
    returncode = 1
    gpu_energy_wh_zeus = None
    zeus_error = None
    try:
        if zeus is not None:
            with zeus:
                proc = subprocess.run(command, cwd=args.working_dir)
                returncode = int(proc.returncode)
            gpu_energy_wh_zeus, zeus_error = zeus.end()
        else:
            proc = subprocess.run(command, cwd=args.working_dir)
            returncode = int(proc.returncode)
    finally:
        end = time.time()
        sampler.stop()

    rapl_after = read_rapl_packages()
    wall_seconds = end - start
    gpu_energy_wh_smi = sampler.energy_wh(wall_seconds)
    gpu_energy_wh = gpu_energy_wh_zeus if gpu_energy_wh_zeus is not None else gpu_energy_wh_smi
    gpu_method = "zeus" if gpu_energy_wh_zeus is not None else ("nvidia_smi" if gpu_energy_wh_smi is not None else "unavailable")
    cpu_energy_wh = rapl_delta_uj(rapl_before, rapl_after) / 3_600_000_000.0 if rapl_before and rapl_after else None

    payload = {
        "label": args.label,
        "command": command,
        "working_dir": args.working_dir or os.getcwd(),
        "returncode": returncode,
        "wall_seconds": wall_seconds,
        "gpu_energy_Wh": gpu_energy_wh,
        "gpu_energy_method": gpu_method,
        "gpu_energy_Wh_zeus": gpu_energy_wh_zeus,
        "gpu_energy_Wh_nvidia_smi": gpu_energy_wh_smi,
        "zeus_error": zeus_error,
        "cpu_energy_Wh": cpu_energy_wh,
        "cpu_energy_method": "intel_rapl" if cpu_energy_wh is not None else "unavailable",
        "total_energy_Wh": (gpu_energy_wh or 0.0) + (cpu_energy_wh or 0.0),
        "start_time": start,
        "end_time": end,
        "nvidia_smi_samples": len(sampler.samples),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[energy] saved measurement: {output_path}")
    print(f"[energy] wall_seconds={wall_seconds:.2f} gpu_Wh={gpu_energy_wh} cpu_Wh={cpu_energy_wh} total_Wh={payload['total_energy_Wh']}")
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
