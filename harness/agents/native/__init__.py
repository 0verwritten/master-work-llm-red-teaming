from __future__ import annotations

__all__ = ["NativeRunResult", "NativeRunner"]


def __getattr__(name: str) -> object:
    if name in __all__:
        from harness.agents.native.runner import NativeRunner, NativeRunResult

        return {
            "NativeRunResult": NativeRunResult,
            "NativeRunner": NativeRunner,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
