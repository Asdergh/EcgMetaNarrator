"""scripts/gpu_monitor.py — Шаг 17 плана.

Опрашивает GPU через pynvml в отдельном потоке и пишет временной ряд в CSV:
  timestamp_iso, timestamp_unix, run_id, concurrency,
  gpu_index, memory_used_mb, memory_total_mb, memory_used_pct,
  utilization_gpu_pct, utilization_mem_pct, temperature_c, power_w

Запускается на ТОЙ ЖЕ машине, что и vLLM (иначе метрики не про этот GPU).

Использование:
  python scripts/gpu_monitor.py --run-id smoke --concurrency 1 --out results/smoke/gpu_timeseries.csv --duration-s 60
  python scripts/gpu_monitor.py --run-id smoke --concurrency 1 --out ... # без --duration-s: до Ctrl+C
"""

from __future__ import annotations

import argparse
import csv
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

FIELDS = [
    "timestamp_iso",
    "timestamp_unix",
    "run_id",
    "concurrency",
    "gpu_index",
    "memory_used_mb",
    "memory_total_mb",
    "memory_used_pct",
    "utilization_gpu_pct",
    "utilization_mem_pct",
    "temperature_c",
    "power_w",
]


class GpuMonitor:
    def __init__(self, out_path: Path, run_id: str, concurrency: int, interval_ms: int = 300):
        self.out_path = out_path
        self.run_id = run_id
        self.concurrency = concurrency
        self.interval_s = interval_ms / 1000.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples = 0

    def _loop(self, nvml, handles) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            while not self._stop.is_set():
                now = datetime.now(timezone.utc)
                ts_unix = now.timestamp()
                for idx, h in enumerate(handles):
                    mem = nvml.nvmlDeviceGetMemoryInfo(h)
                    try:
                        util = nvml.nvmlDeviceGetUtilizationRates(h)
                        util_gpu, util_mem = util.gpu, util.memory
                    except Exception:  # noqa: BLE001
                        util_gpu, util_mem = None, None
                    try:
                        temp = nvml.nvmlDeviceGetTemperature(h, nvml.NVML_TEMPERATURE_GPU)
                    except Exception:  # noqa: BLE001
                        temp = None
                    try:
                        power = nvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # mW -> W
                    except Exception:  # noqa: BLE001
                        power = None

                    w.writerow(
                        {
                            "timestamp_iso": now.isoformat(timespec="milliseconds"),
                            "timestamp_unix": f"{ts_unix:.3f}",
                            "run_id": self.run_id,
                            "concurrency": self.concurrency,
                            "gpu_index": idx,
                            "memory_used_mb": round(mem.used / (1024**2), 1),
                            "memory_total_mb": round(mem.total / (1024**2), 1),
                            "memory_used_pct": round(100.0 * mem.used / mem.total, 2),
                            "utilization_gpu_pct": util_gpu,
                            "utilization_mem_pct": util_mem,
                            "temperature_c": temp,
                            "power_w": round(power, 1) if power is not None else None,
                        }
                    )
                f.flush()
                self.samples += 1
                self._stop.wait(self.interval_s)

    def start(self) -> None:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            raise RuntimeError("NVML не нашёл ни одного GPU")
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
        self._nvml = pynvml
        self._thread = threading.Thread(target=self._loop, args=(pynvml, handles), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        try:
            self._nvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval-ms", type=int, default=300)
    ap.add_argument("--duration-s", type=float, default=None)
    args = ap.parse_args()

    mon = GpuMonitor(Path(args.out), args.run_id, args.concurrency, args.interval_ms)

    def _handle(_sig, _frm):
        mon.stop()
        print(f"\nstopped; samples={mon.samples}; out={args.out}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    mon.start()
    print(f"monitoring GPU -> {args.out} (interval={args.interval_ms}ms)", file=sys.stderr)
    try:
        if args.duration_s is not None:
            time.sleep(args.duration_s)
            mon.stop()
            print(f"done; samples={mon.samples}; out={args.out}")
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        _handle(None, None)


if __name__ == "__main__":
    main()
