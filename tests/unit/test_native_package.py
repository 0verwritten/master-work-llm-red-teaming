from __future__ import annotations

import importlib
import sys


def test_native_package_does_not_eagerly_import_persistence_stack() -> None:
    for module_name in tuple(sys.modules):
        if module_name == "harness.agents.native" or module_name.startswith("harness.persistence"):
            sys.modules.pop(module_name)

    native = importlib.import_module("harness.agents.native")

    assert "harness.persistence" not in sys.modules
    assert native.NativeRunner.__name__ == "NativeRunner"
