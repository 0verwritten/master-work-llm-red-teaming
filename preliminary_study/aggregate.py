from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from preliminary_study.analyze import analyze
from preliminary_study.trace import VAULT_RUNS, write_json_atomic


def aggregate(run_dirs: list[Path]) -> dict[str, Any]:
    per_model: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        result = analyze(run_dir)
        per_model.append(result)

    answerable = [r for r in per_model if "primary_image_vs_text_carrier" in r["contrasts"]]
    channel_rows = []
    for result in answerable:
        primary = result["contrasts"]["primary_image_vs_text_carrier"]["not_refused"]
        channel_rows.append(
            {
                "model_key": result["model_key"],
                "served_model": result["served_model"],
                "image_rate": primary["mean_rate_first"],
                "text_carrier_rate": primary["mean_rate_second"],
                "mean_paired_difference": primary["mean_paired_difference"],
                "bootstrap_ci": [primary["bootstrap_ci_low"], primary["bootstrap_ci_high"]],
                "sign_test_p": primary["sign_test_p"],
                "direction": result["conclusion"]["direction"],
            }
        )

    agreement_rows = [
        {
            "model_key": result["model_key"],
            "served_model": result["served_model"],
            "supports_vision": result["supports_vision"],
            "n": result["instrument_agreement"]["n"],
            "agreement": result["instrument_agreement"]["agreement"],
            "kappa": result["instrument_agreement"]["kappa"],
        }
        for result in per_model
    ]

    excluded = [
        {
            "model_key": result["model_key"],
            "served_model": result["served_model"],
            "reason": result["vision_note"],
        }
        for result in per_model
        if "primary_image_vs_text_carrier" not in result["contrasts"]
    ]

    return {
        "models_analysed": len(per_model),
        "models_answering_channel_question": len(answerable),
        "channel_question": {
            "criterion": "not_refused",
            "contrast": "image vs text_carrier",
            "per_model": channel_rows,
            "consistent_direction": _consistent_direction(channel_rows),
        },
        "excluded_from_channel_question": excluded,
        "instrument_agreement": {"per_model": agreement_rows},
        "per_model_detail": per_model,
    }


def _consistent_direction(rows: list[dict[str, Any]]) -> str:

    if not rows:
        return "no target could answer the channel question"
    significant = [row for row in rows if row["sign_test_p"] < 0.05]
    if not significant:
        return "no target showed a significant channel difference"

    def label(positive: bool) -> str:
        return "image more permissive" if positive else "image less permissive"

    signs = {row["mean_paired_difference"] > 0 for row in significant}
    if len(signs) > 1:
        return "significant targets disagree in direction"

    headline = label(signs.pop())
    others = [row for row in rows if row["sign_test_p"] >= 0.05]
    opposed = [
        row
        for row in others
        if (row["mean_paired_difference"] > 0) != (significant[0]["mean_paired_difference"] > 0)
    ]
    detail = f"{len(significant)} of {len(rows)} targets significant: {headline}"
    if opposed:
        names = ", ".join(str(row["model_key"]) for row in opposed)
        detail += f"; but {names} trends the opposite way (not significant)"
    return detail


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise several model runs together")
    parser.add_argument("--run-id", action="append", required=True, help="repeatable")
    parser.add_argument("--root", type=Path, default=VAULT_RUNS)
    parser.add_argument("--out", type=Path, help="write the aggregate JSON here")
    args = parser.parse_args()

    result = aggregate([args.root / run_id for run_id in args.run_id])
    if args.out:
        write_json_atomic(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "per_model_detail"}, indent=2))


if __name__ == "__main__":
    main()
