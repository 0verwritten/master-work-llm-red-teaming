from __future__ import annotations

import argparse

import pytest

from scripts.watch_vast_batch import Arm, parse_arm


def test_parse_arm_binds_pid_and_instance_allow_list() -> None:
    assert parse_arm("en:123:456,789") == Arm("en", 123, (456, 789))


@pytest.mark.parametrize("value", ["", "en", "en:x:1", "en:1:", "en:1:0"])
def test_parse_arm_rejects_unsafe_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_arm(value)
