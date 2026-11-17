"""Run the judge-behaviour instrument over a frozen study trace.

Judge rationales live in `llm_calls.response_json`, not in `evaluations`, because
the schema allows only one stored verdict per round (`UniqueConstraint("round_id")`)
and the re-judge passes could not be persisted beside the original. Reading the
call trace recovers every pass: the fixed judge, the graded-rubric re-judge, and
the second judge.

Each call is bound to a language two ways. The original judge is reached through
the round that owns its step; a re-judge is reached through its run's benchmark
item, which the re-judge checkpoint keys by language. Both are exact joins - the
script fails rather than guessing if either stops resolving.

The store holds raw rationale text. This script never prints it: the report
carries counts, rates, and word lengths only.

    .venv/bin/python scripts/analyze_judge_behaviour.py \
        --store .translation-workspace/half-study.sqlite3 \
        --rejudge-checkpoint .translation-workspace/rejudge-checkpoint.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from harness.analysis.judge_behaviour import (
    JudgeResponse,
    analyse,
    disagreement_profile,
    label_mix,
)

FENCE_OPEN = re.compile(r"^```[a-zA-Z]*\s*")
FENCE_CLOSE = re.compile(r"\s*```$")


def parse_verdict(content: Any) -> dict[str, Any] | None:
    if not isinstance(content, str):
        return None
    text = content.strip()
    if text.startswith("```"):
        text = FENCE_CLOSE.sub("", FENCE_OPEN.sub("", text))
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def pass_name(purpose: str, model: str) -> str:
    if purpose == "judge":
        return "fixed judge (GPT-OSS 20B)"
    if "gemma" in model.lower():
        return "second judge (Gemma 4 31B)"
    return "graded re-judge (GPT-OSS 20B)"


def load(store: Path, checkpoint: Path) -> tuple[list[JudgeResponse], dict[str, Any]]:
    connection = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row

    language_of_item: dict[str, str] = {}
    hash_to_items: dict[str, set[str]] = defaultdict(set)
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        item, _, response_hash = str(record["cell_id"]).partition(":")
        language_of_item[item] = str(record["language"])
        if response_hash:
            hash_to_items[response_hash].add(item)

    language_of_step = {
        row["step_id"]: row["language"]
        for row in connection.execute(
            "select ar.judge_step_id as step_id, s.language as language "
            "from adversarial_rounds ar "
            "join adversarial_sessions s on s.id = ar.session_id "
            "where ar.judge_step_id is not null"
        )
    }

    responses: list[JudgeResponse] = []
    skipped: Counter[str] = Counter()
    for row in connection.execute(
        "select l.purpose, l.model, l.step_id, l.response_json, "
        "       r.benchmark_item_id as item "
        "from llm_calls l join runs r on r.id = l.run_id "
        "where l.purpose in ('judge', 'rejudge') and l.status = 'succeeded'"
    ):
        verdict = parse_verdict(json.loads(row["response_json"] or "{}").get("content"))
        if verdict is None:
            skipped["unparseable"] += 1
            continue
        rationale = verdict.get("rationale")
        label = verdict.get("label")
        if not isinstance(rationale, str) or not rationale.strip():
            skipped["no rationale"] += 1
            continue
        language = language_of_step.get(row["step_id"]) or language_of_item.get(row["item"])
        if language is None:
            skipped["no language"] += 1
            continue
        confidence = verdict.get("confidence")
        responses.append(
            JudgeResponse(
                judge_pass=pass_name(str(row["purpose"]), str(row["model"])),
                language=str(language),
                label=label,
                confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
                rationale=rationale,
            )
        )

    second: dict[str, list[str]] = defaultdict(list)
    for row in connection.execute(
        "select l.response_json, r.benchmark_item_id as item "
        "from llm_calls l join runs r on r.id = l.run_id "
        "where l.purpose = 'rejudge' and l.model like '%gemma%' and l.status = 'succeeded'"
    ):
        verdict = parse_verdict(json.loads(row["response_json"] or "{}").get("content"))
        if verdict and verdict.get("label"):
            second[row["item"]].append(str(verdict["label"]))

    agreed: list[JudgeResponse] = []
    disagreed: list[JudgeResponse] = []
    flips: Counter[tuple[str, str, str]] = Counter()
    for row in connection.execute(
        "select ar.target_response as response, s.language as language, "
        "       e.label as label, e.confidence as confidence, e.rationale as rationale "
        "from adversarial_rounds ar "
        "join adversarial_sessions s on s.id = ar.session_id "
        "join evaluations e on e.round_id = ar.id"
    ):
        digest = hashlib.sha256((row["response"] or "").encode()).hexdigest()[:8]
        labels = [label for item in hash_to_items.get(digest, ()) for label in second.get(item, ())]
        if not labels:
            continue
        other = Counter(labels).most_common(1)[0][0]
        observation = JudgeResponse(
            judge_pass="fixed judge (GPT-OSS 20B)",
            language=str(row["language"]),
            label=row["label"],
            confidence=row["confidence"],
            rationale=str(row["rationale"] or "-"),
        )
        if other == row["label"]:
            agreed.append(observation)
        else:
            disagreed.append(observation)
            flips[(str(row["language"]), str(row["label"]), other)] += 1

    connection.close()
    context = {
        "skipped": dict(skipped),
        "agreed": agreed,
        "disagreed": disagreed,
        "flips": flips,
    }
    return responses, context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True, help="study SQLite store")
    parser.add_argument(
        "--rejudge-checkpoint",
        type=Path,
        required=True,
        help="re-judge checkpoint JSONL, which carries the language per cell",
    )
    arguments = parser.parse_args()
    for path in (arguments.store, arguments.rejudge_checkpoint):
        if not path.exists():
            raise SystemExit(f"no such file: {path}")

    responses, context = load(arguments.store, arguments.rejudge_checkpoint)
    if not responses:
        raise SystemExit("no judge responses could be read from this store")

    by_stratum: dict[tuple[str, str], list[JudgeResponse]] = defaultdict(list)
    for value in responses:
        by_stratum[(value.judge_pass, value.language)].append(value)

    payload: dict[str, Any] = analyse(responses).as_json()
    for stratum in payload["strata"]:
        key = (stratum["judge_pass"], stratum["language"])
        stratum["label_mix"] = label_mix(by_stratum[key])
    payload["skipped_calls"] = context["skipped"]

    agreed, disagreed = context["agreed"], context["disagreed"]
    if agreed and disagreed:
        payload["cross_judge_confidence"] = disagreement_profile(agreed, disagreed).as_json()
        payload["disagreement_directions"] = {
            f"{language}: fixed={mine} second={theirs}": count
            for (language, mine, theirs), count in sorted(
                context["flips"].items(), key=lambda item: -item[1]
            )
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
