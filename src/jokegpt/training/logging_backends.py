"""Where training metrics go.

The notebooks called Comet directly with an API key pasted into the cell. These
loggers share one interface, and the Comet one reads its credentials from the
environment, so no key ever lands in source.
"""

from __future__ import annotations

import os
from typing import Protocol


class MetricLogger(Protocol):
    def log_parameters(self, params: dict) -> None: ...
    def log_metric(self, name: str, value: float, step: int) -> None: ...
    def end(self) -> None: ...


class ConsoleLogger:
    """Default logger: prints to stdout. No accounts, no network."""

    def log_parameters(self, params: dict) -> None:
        print("hyperparameters:")
        for key in sorted(params):
            print(f"  {key}: {params[key]}")

    def log_metric(self, name: str, value: float, step: int) -> None:
        print(f"  step {step:>6} | {name}: {value:.4f}")

    def end(self) -> None:
        pass


class CometLogger:
    """Optional Comet ML backend.

    Requires `comet_ml` installed and these environment variables:
      COMET_API_KEY, COMET_PROJECT_NAME, COMET_WORKSPACE
    """

    REQUIRED_ENV = ("COMET_API_KEY", "COMET_PROJECT_NAME", "COMET_WORKSPACE")

    def __init__(self) -> None:
        missing = [name for name in self.REQUIRED_ENV if not os.environ.get(name)]
        if missing:
            raise RuntimeError(
                f"Comet logging requires {', '.join(missing)} in the environment. "
                "Copy .env.example and fill it in, or use the console logger."
            )
        try:
            from comet_ml import Experiment
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "comet_ml is not installed. Run: pip install comet_ml"
            ) from exc

        self._experiment = Experiment(
            api_key=os.environ["COMET_API_KEY"],
            project_name=os.environ["COMET_PROJECT_NAME"],
            workspace=os.environ["COMET_WORKSPACE"],
        )

    def log_parameters(self, params: dict) -> None:
        self._experiment.log_parameters(params)

    def log_metric(self, name: str, value: float, step: int) -> None:
        self._experiment.log_metric(name, value, epoch=step)

    def end(self) -> None:
        self._experiment.end()


def get_logger(backend: str) -> MetricLogger:
    """Build a logger by name: 'console' or 'comet'."""
    backends = {"console": ConsoleLogger, "comet": CometLogger}
    if backend not in backends:
        raise ValueError(f"unknown logger {backend!r}; expected {sorted(backends)}")
    return backends[backend]()
