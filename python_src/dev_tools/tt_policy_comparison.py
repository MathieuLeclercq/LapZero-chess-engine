"""Compare plusieurs resultats puzzle sans relancer la recherche."""

import sys as _runtime_sys
from pathlib import Path as _RuntimePath
if __package__ in (None, ""):
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
import _runtime


import argparse
import csv
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

from evaluation.bench_metrics import mcnemar


@dataclass(frozen=True)
class PolicySummary:
    name: str
    total: int
    solved: int
    solve_rate: float
    duration_s: float


@dataclass(frozen=True)
class PairSummary:
    left: str
    right: str
    left_only: int
    right_only: int
    move_agreement: float
    solve_rate_delta: float
    mcnemar_p: float


@dataclass(frozen=True)
class PolicyComparison:
    common_lines: int
    policies: tuple[PolicySummary, ...]
    pairs: tuple[PairSummary, ...]


def _bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"booleen invalide : {value!r}")


def _read_csv(path: Path) -> dict[int, dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        rows: dict[int, dict[str, str]] = {}
        for row in csv.DictReader(f):
            try:
                line = int(row["ligne"])
                _bool(row["reussi_recherche"])
                float(row["duree_s"])
                row["coup_recherche"]
                row["erreur"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"CSV invalide {path}: {exc}") from exc
            if line in rows:
                raise ValueError(f"ligne dupliquee {line} dans {path}")
            rows[line] = row
    return rows


def compare_policy_csv(inputs: dict[str, Path]) -> PolicyComparison:
    if not inputs:
        raise ValueError("aucune politique a comparer")

    datasets = {name: _read_csv(Path(path)) for name, path in inputs.items()}
    common = set.intersection(*(set(rows) for rows in datasets.values()))
    valid_lines = sorted(
        line for line in common
        if all(not rows[line]["erreur"] for rows in datasets.values()))

    policies = []
    for name, rows in datasets.items():
        solved = sum(_bool(rows[line]["reussi_recherche"])
                     for line in valid_lines)
        duration = sum(float(rows[line]["duree_s"]) for line in valid_lines)
        total = len(valid_lines)
        policies.append(PolicySummary(
            name=name, total=total, solved=solved,
            solve_rate=solved / total if total else 0.0,
            duration_s=duration))

    pairs = []
    for left, right in itertools.combinations(datasets, 2):
        left_rows = datasets[left]
        right_rows = datasets[right]
        left_only = right_only = agreements = 0
        for line in valid_lines:
            left_solved = _bool(left_rows[line]["reussi_recherche"])
            right_solved = _bool(right_rows[line]["reussi_recherche"])
            left_only += int(left_solved and not right_solved)
            right_only += int(right_solved and not left_solved)
            agreements += int(left_rows[line]["coup_recherche"]
                              == right_rows[line]["coup_recherche"])
        _, p = mcnemar(left_only, right_only)
        total = len(valid_lines)
        pairs.append(PairSummary(
            left=left, right=right,
            left_only=left_only, right_only=right_only,
            move_agreement=agreements / total if total else 0.0,
            solve_rate_delta=(right_only - left_only) / total if total else 0.0,
            mcnemar_p=p))

    return PolicyComparison(
        common_lines=len(valid_lines),
        policies=tuple(policies), pairs=tuple(pairs))


def format_report(comparison: PolicyComparison) -> str:
    lines = [
        "# Comparaison des politiques TT",
        "",
        f"Puzzles communs valides : **{comparison.common_lines}**",
        "",
        "## Resultats par politique",
        "",
        "| Politique | n | Resolus | Reussite | Duree cumulee |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in comparison.policies:
        lines.append(
            f"| {policy.name} | {policy.total} | {policy.solved} "
            f"| {100 * policy.solve_rate:.2f} % | {policy.duration_s:.1f} s |")

    lines += [
        "",
        "## Comparaisons appariees",
        "",
        "Le delta est le taux de droite moins le taux de gauche.",
        "",
        "| Gauche | Droite | Gauche seule | Droite seule | Accord coups "
        "| Delta reussite | McNemar p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for pair in comparison.pairs:
        lines.append(
            f"| {pair.left} | {pair.right} | {pair.left_only} "
            f"| {pair.right_only} | {100 * pair.move_agreement:.2f} % "
            f"| {100 * pair.solve_rate_delta:+.2f} points "
            f"| {pair.mcnemar_p:.3g} |")
    return "\n".join(lines) + "\n"


def _parse_inputs(values: list[str]) -> dict[str, Path]:
    inputs = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"entree attendue sous la forme nom=chemin : {value}")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path or name in inputs:
            raise ValueError(f"entree invalide ou dupliquee : {value}")
        inputs[name] = Path(raw_path)
    return inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True,
                        help="resultats nommes, par exemple legacy=a.csv h0=b.csv")
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()

    try:
        comparison = compare_policy_csv(_parse_inputs(args.input))
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(format_report(comparison), encoding="utf-8")
    print(f"Rapport : {args.out_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
