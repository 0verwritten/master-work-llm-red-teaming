"""Run the attacker-dynamics instrument over a frozen study trace.

Reads a study SQLite store from the Git-ignored `.translation-workspace/` tree
and prints the aggregate report as JSON. The store holds raw attack and response
text; this script never prints any of it, because `AttackerDynamicsReport`
carries counts and rates only.

    .venv/bin/python scripts/analyze_attacker_dynamics.py \
        --store .translation-workspace/half-study.sqlite3
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from harness.analysis.attacker_dynamics import RoundObservation, analyse

QUERY = """
select
    ar.session_id as session_id,
    ar.round_number as round_number,
    s.language as language,
    ar.attack_text as attack_text,
    e.label as verdict
from adversarial_rounds ar
join adversarial_sessions s on s.id = ar.session_id
left join evaluations e on e.round_id = ar.id
order by ar.session_id, ar.round_number
"""


def load(store: Path) -> list[RoundObservation]:
    connection = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return [
            RoundObservation(
                session_id=str(row["session_id"]),
                round_number=int(row["round_number"]),
                language=str(row["language"]),
                attack_text=str(row["attack_text"] or ""),
                verdict=row["verdict"],
            )
            for row in connection.execute(QUERY)
        ]
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True, help="study SQLite store")
    parser.add_argument("--rounds", type=int, default=3, help="rounds a complete session runs")
    arguments = parser.parse_args()
    if not arguments.store.exists():
        raise SystemExit(f"no such store: {arguments.store}")
    report = analyse(load(arguments.store), rounds_required=arguments.rounds)
    print(json.dumps(report.as_json(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
