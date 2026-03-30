import os
import threading
import time
from pathlib import Path


def _safe_read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _safe_read_int(path):
    text = _safe_read_text(path)
    if text is None:
        return None
    try:
        return int(text)
    except Exception:
        return None


def discover_rapl_energy_sources():
    powercap_root = Path("/sys/class/powercap")
    if not powercap_root.exists():
        return []

    sources = []
    for energy_path in powercap_root.rglob("energy_uj"):
        source_dir = energy_path.parent
        max_range = _safe_read_int(source_dir / "max_energy_range_uj")
        name = _safe_read_text(source_dir / "name") or source_dir.name
        if max_range is None:
            continue
        sources.append(
            {
                "name": name,
                "energy_path": energy_path,
                "max_range_uj": int(max_range),
            }
        )
    return sources


def take_rapl_snapshot(sources):
    snapshot = {}
    for source in sources:
        value = _safe_read_int(source["energy_path"])
        if value is not None:
            snapshot[str(source["energy_path"])] = int(value)
    return snapshot


def compute_rapl_delta_wh(sources, start_snapshot, end_snapshot):
    total_uj = 0
    for source in sources:
        key = str(source["energy_path"])
        start_value = start_snapshot.get(key)
        end_value = end_snapshot.get(key)
        if start_value is None or end_value is None:
            continue
        delta = int(end_value) - int(start_value)
        if delta < 0:
            delta += int(source["max_range_uj"])
        total_uj += max(delta, 0)
    return float(total_uj) / 3.6e9


class NvmlEnergyPoller:
    def __init__(self, poll_interval_ms=200):
        self.poll_interval_s = max(float(poll_interval_ms) / 1000.0, 0.05)
        self._nvml = None
        self._handles = None
        self._thread = None
        self._stop_event = threading.Event()
        self._energy_ws = 0.0
        self._last_time = None
        self._available = False
        self._error = None

        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            count = int(pynvml.nvmlDeviceGetCount())
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
            self._nvml = pynvml
            self._available = count > 0
        except Exception as exc:
            self._error = str(exc)
            self._available = False

    @property
    def available(self):
        return bool(self._available)

    @property
    def error(self):
        return self._error

    def _sample_power_w(self):
        if not self._available:
            return 0.0
        total_w = 0.0
        for handle in self._handles:
            total_w += float(self._nvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
        return total_w

    def _run(self):
        self._last_time = time.perf_counter()
        while not self._stop_event.wait(self.poll_interval_s):
            current_time = time.perf_counter()
            power_w = self._sample_power_w()
            self._energy_ws += power_w * max(current_time - self._last_time, 0.0)
            self._last_time = current_time

    def start(self):
        if not self._available:
            return
        self._stop_event.clear()
        self._energy_ws = 0.0
        self._thread = threading.Thread(target=self._run, name="nvml-energy-poller", daemon=True)
        self._thread.start()

    def stop(self):
        if not self._available or self._thread is None:
            return 0.0
        current_time = time.perf_counter()
        power_w = self._sample_power_w()
        if self._last_time is not None:
            self._energy_ws += power_w * max(current_time - self._last_time, 0.0)
        self._stop_event.set()
        self._thread.join(timeout=max(self.poll_interval_s * 2.0, 1.0))
        self._thread = None
        return float(self._energy_ws) / 3600.0

    def close(self):
        try:
            if self._nvml is not None:
                self._nvml.nvmlShutdown()
        except Exception:
            pass


class EnergyMeter:
    def __init__(self, backend="auto", poll_interval_ms=200):
        self.backend = backend
        self.poll_interval_ms = int(poll_interval_ms)
        self.cpu_energy_wh = None
        self.gpu_energy_wh = None
        self.total_energy_wh = 0.0
        self.cpu_backend = None
        self.gpu_backend = None
        self.cpu_available = False
        self.gpu_available = False
        self._rapl_sources = []
        self._rapl_start = None
        self._nvml_poller = None
        self._zeus_monitor = None
        self._zeus_window = None
        self._zeus_available = False

    def _enable_cpu_rapl(self):
        self._rapl_sources = discover_rapl_energy_sources()
        if not self._rapl_sources:
            return False
        self._rapl_start = take_rapl_snapshot(self._rapl_sources)
        self.cpu_backend = "rapl"
        self.cpu_available = True
        return True

    def _enable_gpu_zeus(self):
        if self.backend not in ("auto", "zeus"):
            return False
        try:
            from zeus.monitor import ZeusMonitor  # type: ignore

            self._zeus_monitor = ZeusMonitor()
            self._zeus_window = f"selection_{os.getpid()}_{int(time.time() * 1000)}"
            self._zeus_monitor.begin_window(self._zeus_window)
            self.gpu_backend = "zeus"
            self.gpu_available = True
            self._zeus_available = True
            return True
        except Exception:
            self._zeus_monitor = None
            self._zeus_window = None
            self._zeus_available = False
            return False

    def _enable_gpu_nvml(self):
        if self.backend not in ("auto", "nvml"):
            return False
        self._nvml_poller = NvmlEnergyPoller(poll_interval_ms=self.poll_interval_ms)
        if not self._nvml_poller.available:
            self._nvml_poller.close()
            self._nvml_poller = None
            return False
        self._nvml_poller.start()
        self.gpu_backend = "nvml"
        self.gpu_available = True
        return True

    def start(self):
        if self.backend in ("auto", "rapl"):
            self._enable_cpu_rapl()
        if self.backend != "none":
            if not self._enable_gpu_zeus():
                self._enable_gpu_nvml()
        return self

    def stop(self):
        if self.cpu_available and self._rapl_sources and self._rapl_start is not None:
            rapl_end = take_rapl_snapshot(self._rapl_sources)
            self.cpu_energy_wh = compute_rapl_delta_wh(self._rapl_sources, self._rapl_start, rapl_end)

        if self._zeus_available and self._zeus_monitor is not None and self._zeus_window is not None:
            try:
                measurement = self._zeus_monitor.end_window(self._zeus_window)
                gpu_joules = None
                for attr_name in ("gpu_energy", "total_gpu_energy", "energy"):
                    if hasattr(measurement, attr_name):
                        gpu_joules = getattr(measurement, attr_name)
                        break
                if gpu_joules is not None:
                    self.gpu_energy_wh = float(gpu_joules) / 3600.0
            except Exception:
                self.gpu_energy_wh = None

        if self._nvml_poller is not None:
            try:
                self.gpu_energy_wh = self._nvml_poller.stop()
            finally:
                self._nvml_poller.close()
                self._nvml_poller = None

        total = 0.0
        if self.cpu_energy_wh is not None:
            total += float(self.cpu_energy_wh)
        if self.gpu_energy_wh is not None:
            total += float(self.gpu_energy_wh)
        self.total_energy_wh = float(total)
        return self

    def to_dict(self):
        return {
            "cpu_energy_wh": None if self.cpu_energy_wh is None else float(self.cpu_energy_wh),
            "gpu_energy_wh": None if self.gpu_energy_wh is None else float(self.gpu_energy_wh),
            "total_energy_wh": float(self.total_energy_wh),
            "cpu_backend": self.cpu_backend,
            "gpu_backend": self.gpu_backend,
            "cpu_available": bool(self.cpu_available),
            "gpu_available": bool(self.gpu_available),
            "backend": self.backend,
        }

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False
