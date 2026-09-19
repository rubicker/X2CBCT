from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModuleStatus:
    name: str
    available: bool
    version: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class EnvironmentReport:
    project_root: Path
    python_executable: str
    expected_python: Path
    in_project_env: bool
    modules: dict[str, ModuleStatus]

    @property
    def ok(self) -> bool:
        return self.in_project_env and all(item.available for item in self.modules.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "project_root": str(self.project_root),
            "python_executable": self.python_executable,
            "expected_python": str(self.expected_python),
            "in_project_env": self.in_project_env,
            "ok": self.ok,
            "modules": {
                name: {
                    "available": status.available,
                    "version": status.version,
                    "error": status.error,
                }
                for name, status in self.modules.items()
            },
        }


def _module_status(name: str) -> ModuleStatus:
    if importlib.util.find_spec(name) is None:
        return ModuleStatus(name=name, available=False, error="module not found")
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", None)
        if name == "SimpleITK":
            version = module.Version_VersionString()
        if name == "cv2":
            version = module.__version__
        return ModuleStatus(name=name, available=True, version=str(version) if version else None)
    except Exception as exc:  # pragma: no cover - exercised only on broken installs
        return ModuleStatus(name=name, available=False, error=repr(exc))


def check_environment(project_root: str | Path) -> EnvironmentReport:
    root = Path(project_root).resolve()
    expected_python = root / ".conda" / "x2cbct-v0" / "python.exe"
    executable = str(Path(sys.executable).resolve())
    expected = str(expected_python.resolve())
    module_names = ("torch", "numpy", "scipy", "SimpleITK", "yaml", "pytest", "cv2", "skimage", "tqdm")
    return EnvironmentReport(
        project_root=root,
        python_executable=executable,
        expected_python=expected_python,
        in_project_env=executable.lower() == expected.lower(),
        modules={name: _module_status(name) for name in module_names},
    )
