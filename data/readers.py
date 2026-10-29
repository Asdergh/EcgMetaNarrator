"""Signal readers. Loaders intentionally do not call this module."""

from pathlib import Path
from typing import Any


def _header_metadata(header_path: Path) -> dict[str, Any]:
    first_line = header_path.read_text(encoding="ascii", errors="replace").splitlines()[0]
    fields = first_line.split()
    if len(fields) < 3:
        raise ValueError(f"Invalid WFDB header: {header_path}")
    fs_field = fields[2].split("/")[0]
    return {
        "n_sig": int(fields[1]),
        "fs": float(fs_field),
    }


class SignalReader:
    """Считывание зарегистрированной записи из файла, отдельно обработав различия между форматами"""

    @staticmethod
    def read(file_path: str | Path) -> Any:
        path = Path(file_path)
        mat_path = path.with_suffix(".mat")
        hea_path = path.with_suffix(".hea")
        if mat_path.exists():
            try:
                from scipy.io import loadmat
            except ImportError as exc:
                raise ImportError("Reading .mat records requires scipy") from exc
            return loadmat(mat_path)
        if hea_path.exists() and path.with_suffix(".dat").exists():
            try:
                import wfdb
            except ImportError as exc:
                raise ImportError("Reading WFDB records requires wfdb") from exc
            return wfdb.rdrecord(str(path))
        raise FileNotFoundError(f"No .mat or .hea record found for {path}")

    @staticmethod
    def metadata(file_path: str | Path) -> dict[str, Any]:
        path = Path(file_path)
        hea_path = path.with_suffix(".hea")
        if hea_path.exists() and path.with_suffix(".dat").exists():
            return _header_metadata(hea_path)
        if path.with_suffix(".mat").exists():
            return {"n_sig": None, "fs": None}
        raise FileNotFoundError(f"No .mat or .hea record found for {path}")
